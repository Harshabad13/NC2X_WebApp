import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile
from streamlit_agraph import agraph, Node, Edge, Config
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- PAGE CONFIG ---
st.set_page_config(page_title="NC2X Project", layout="wide", page_icon="🧠")
st.title("🧠 NC2X: Concept, Causal & Context-Aware AI")

# --- SETTINGS ---
MODEL_PATH = 'models/nc2x_model_epoch_30.pth'
DEVICE = torch.device('cpu') 

# --- COCO CLASSES ---
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# --- MODEL ARCHITECTURE ---
class NC2X_Model(nn.Module):
    def __init__(self, num_classes=80, feature_dim=2048, hidden_dim=512):
        super(NC2X_Model, self).__init__()
        try: resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except: resnet = models.resnet50(pretrained=True)
        self.concept_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.gnn1 = GCNConv(feature_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        self.fusion_layer = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.final_predictor = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    def forward(self, image_tensor, graph_batch):
        with torch.no_grad():
            global_features = self.concept_extractor(image_tensor)
            global_features = global_features.view(global_features.size(0), -1)
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = self.relu(self.gnn1(x, edge_index))
        x = self.relu(self.gnn2(x, edge_index))
        graph_features = global_mean_pool(x, batch)
        combined_features = torch.cat([global_features, graph_features], dim=1)
        fused = self.relu(self.fusion_layer(combined_features))
        return self.final_predictor(fused)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    yolo = YOLO('yolov8n.pt')
    try: res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except: res = models.resnet50(pretrained=True)
    
    # Feature Extractor (For GNN)
    extractor = nn.Sequential(*list(res.children())[:-2]).eval().to(DEVICE)
    
    # NC2X Model
    nc2x = NC2X_Model(num_classes=80).to(DEVICE)
    msg = "Initializing..."
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'): k = k[7:]
                new_state_dict[k] = v
            nc2x.load_state_dict(new_state_dict, strict=False)
            msg = "✅ Model Loaded Successfully!"
        except Exception as e:
            msg = f"⚠️ Error loading weights: {e}"
    else:
        msg = f"❌ Model file not found at {MODEL_PATH}"
        
    nc2x.eval()
    return yolo, extractor, nc2x, msg

yolo, extractor, nc2x_model, status_msg = load_resources()

if "Success" in status_msg:
    st.sidebar.success(status_msg)
else:
    st.sidebar.error(status_msg)

# --- HELPER: GRAPH VISUALIZATION ---
def render_knowledge_graph(labels, width=600, height=300):
    nodes = []
    edges = []
    unique_labels = []
    for l in labels:
        if l < len(COCO_CLASSES): unique_labels.append(COCO_CLASSES[l])
    
    if not unique_labels: return

    nodes.append(Node(id="SCENE", label="SCENE", size=25, color="#FF5722", shape="diamond"))
    counts = {}
    for name in unique_labels:
        counts[name] = counts.get(name, 0) + 1
        node_id = f"{name}_{counts[name]}"
        nodes.append(Node(id=node_id, label=name, size=20, color="#03A9F4", shape="dot"))
        edges.append(Edge(source=node_id, target="SCENE", color="#bdc3c7"))
        if len(nodes) > 2:
            prev_node = nodes[-2].id
            edges.append(Edge(source=prev_node, target=node_id, color="#ecf0f1"))

    config = Config(width=width, height=height, directed=False, physics=True, hierarchy=False)
    return agraph(nodes=nodes, edges=edges, config=config)

# --- PREDICTION LOGIC ---
def predict(image):
    results = yolo(image, conf=0.4, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu()
    labels = results.boxes.cls.cpu().int().tolist()

    if len(boxes) == 0:
        node_features = torch.zeros(1, 2048, device=DEVICE)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
    else:
        t_feat = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        with torch.no_grad():
            img_tensor = t_feat(image.convert('RGB')).unsqueeze(0).to(DEVICE)
            f_map = extractor(img_tensor)
            feats = []
            for _ in boxes: feats.append(f_map.mean([2, 3]).squeeze(0))
            node_features = torch.stack(feats)
            num_nodes = len(boxes)
            adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            edge_index = adj.nonzero(as_tuple=False).t().contiguous().to(DEVICE)

    graph_data = Data(x=node_features, edge_index=edge_index)
    batch = Batch.from_data_list([graph_data]).to(DEVICE)
    t_img = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    with torch.no_grad():
        out = nc2x_model(t_img(image.convert('RGB')).unsqueeze(0).to(DEVICE), batch)
        probs = torch.sigmoid(out).flatten().cpu().numpy()
    return probs, boxes, labels

# --- SIDEBAR MENU ---
app_mode = st.sidebar.selectbox("Choose Mode", [
    "Image Analysis", 
    "Video Analysis", 
    "Live Webcam", 
    "Causal Experiment",
    "Comparison (vs Grad-CAM)"
])

# ==========================================
# 1. IMAGE ANALYSIS
# ==========================================
if app_mode == "Image Analysis":
    st.header("🖼️ Image Analysis & Scene Graph")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2, col3 = st.columns([1, 1, 1.5])
        
        with col1: st.image(image, caption="Original", use_container_width=True)
        
        if st.button("Analyze Scene"):
            with st.spinner("Analyzing..."):
                probs, boxes, labels = predict(image)
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = map(int, box)
                    if labels[i] < len(COCO_CLASSES):
                        cv2.rectangle(img_cv, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(img_cv, COCO_CLASSES[labels[i]], (x1, y1-10), 0, 0.9, (36,255,12), 2)
                
                with col2:
                    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="YOLO Detection", use_container_width=True)
                    st.subheader("Top Concepts")
                    for i in np.argsort(probs)[-5:][::-1]:
                        score = float(probs[i])
                        st.write(f"**{COCO_CLASSES[i]}**: {score:.0%}")
                        st.progress(min(score, 1.0))

                with col3:
                    st.subheader("🕸️ Visual Scene Graph")
                    render_knowledge_graph(labels, height=350)

# ==========================================
# 2. VIDEO ANALYSIS
# ==========================================
elif app_mode == "Video Analysis":
    st.header("🎥 Video Analysis")
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        col1, col2 = st.columns([2, 1])
        with col1: stframe = st.empty()
        with col2: 
            st.subheader("Live Stats")
            stats_ph = st.empty()

        if st.button("Start Video"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (640, 360))
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                probs, boxes, labels = predict(pil_img)
                
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = map(int, box)
                    if labels[i] < len(COCO_CLASSES):
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(frame, COCO_CLASSES[labels[i]], (x1, y1-5), 0, 0.5, (0,255,0), 1)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                with stats_ph.container():
                    for i in np.argsort(probs)[-5:][::-1]:
                        score = float(probs[i])
                        if score > 0.2:
                            st.write(f"**{COCO_CLASSES[i]}**: {score:.0%}")
                            st.progress(min(score, 1.0))
        cap.release()

# ==========================================
# 3. LIVE WEBCAM
# ==========================================
elif app_mode == "Live Webcam":
    st.header("📹 Live Webcam")
    col1, col2 = st.columns([2, 1])
    with col1:
        run = st.checkbox('Start Camera')
        FRAME_WINDOW = st.image([])
    with col2:
        st.subheader("Live Stats")
        stats_ph = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret: break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs, boxes, labels = predict(pil_img)
            for i, box in enumerate(boxes):
                x1,y1,x2,y2 = map(int, box)
                if labels[i] < len(COCO_CLASSES):
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, COCO_CLASSES[labels[i]], (x1, y1-5), 0, 0.5, (0,255,0), 1)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with stats_ph.container():
                for i in np.argsort(probs)[-5:][::-1]:
                    score = float(probs[i])
                    if score > 0.2:
                        st.write(f"**{COCO_CLASSES[i]}**: {score:.0%}")
                        st.progress(min(score, 1.0))
        cap.release()

# ==========================================
# 4. CAUSAL EXPERIMENT
# ==========================================
elif app_mode == "Causal Experiment":
    st.header("🧪 Causal Analysis & Graph Logic")
    file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    if file:
        image = Image.open(file).convert('RGB')
        orig_probs, orig_boxes, orig_labels = predict(image)
        detected_names = sorted(list(set([COCO_CLASSES[i] for i in orig_labels if i < len(COCO_CLASSES)])))
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
            st.subheader("🕸️ Original Graph")
            render_knowledge_graph(orig_labels, height=300)
        with col2:
            st.subheader("Intervention")
            remove_name = st.selectbox("Remove Node:", detected_names)
            target_name = st.selectbox("Observe Target:", [n for n in detected_names if n != remove_name])
            if st.button("Run Intervention"):
                remove_idx = COCO_CLASSES.index(remove_name)
                target_idx = COCO_CLASSES.index(target_name)
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                remaining_labels = []
                for i, lbl in enumerate(orig_labels):
                    if lbl == remove_idx:
                        x1,y1,x2,y2 = map(int, orig_boxes[i])
                        cv2.rectangle(img_cv, (x1,y1), (x2,y2), (0,0,0), -1)
                    else:
                        remaining_labels.append(lbl)
                masked_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                new_probs, _, _ = predict(masked_pil)
                orig_score = orig_probs[target_idx]
                new_score = new_probs[target_idx]
                drop = (orig_score - new_score) / orig_score * 100 if orig_score > 0.01 else 0
                st.image(masked_pil, caption=f"Intervened (No {remove_name})", use_container_width=True)
                st.metric(f"{target_name} Confidence", f"{new_score:.4f}", f"-{drop:.2f}% Drop", delta_color="inverse")
                st.subheader("🕸️ Altered Graph")
                render_knowledge_graph(remaining_labels, height=300)

# ==========================================
# 5. COMPARISON WITH GRAD-CAM (FIXED)
# ==========================================
elif app_mode == "Comparison (vs Grad-CAM)":
    st.header("🆚 Comparative Analysis: NC2X vs Traditional AI")
    file = st.file_uploader("Upload Image for Comparison", type=['jpg', 'png'])
    
    if file:
        image = Image.open(file).convert('RGB')
        col1, col2 = st.columns(2)
        with col1: st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Run Comparison"):
            with st.spinner("Generating Explanations..."):
                # 1. Run NC2X
                probs, boxes, labels = predict(image)
                
                # 2. Run Grad-CAM (USING A STANDARD MODEL NOW)
                # Hum ek naya standard ResNet load karenge jo "Traditional AI" ko represent karega
                standard_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE).eval()
                target_layers = [standard_model.layer4[-1]]
                cam = GradCAM(model=standard_model, target_layers=target_layers)
                
                t_cam = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                input_tensor = t_cam(image).unsqueeze(0).to(DEVICE)
                
                # Generate Heatmap (Assuming class index 281 for 'tabby cat' or just max)
                # Hum yahan targets=None rakh rahe hain, toh wo highest probability class uthayega
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                grayscale_cam = grayscale_cam[0, :]
                
                img_np = np.array(image.resize((224, 224))) / 255.0
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                
                st.divider()
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("🔴 Traditional AI (Grad-CAM)")
                    st.image(visualization, caption="Pixel Heatmap Explanation", use_container_width=True)
                    st.error("**Limit:** Just colorful blobs. It doesn't tell 'WHY'.")
                
                with c2:
                    st.subheader("🟢 Our NC2X Model")
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    detected = []
                    for i, box in enumerate(boxes):
                        x1,y1,x2,y2 = map(int, box)
                        if labels[i] < len(COCO_CLASSES):
                            name = COCO_CLASSES[labels[i]]
                            detected.append(name)
                            cv2.rectangle(img_cv, (x1,y1), (x2,y2), (0,255,0), 2)
                            cv2.putText(img_cv, name, (x1, y1-10), 0, 0.9, (36,255,12), 2)
                    
                    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Concept & Graph Explanation", use_container_width=True)
                    
                    detected = list(set(detected))
                    if len(detected) > 1:
                        st.success(f"**Advantage:** NC2X understands relationships between **{', '.join(detected[:3])}**.")
                        render_knowledge_graph(labels, height=200)
                    else:
                        st.success("**Advantage:** Explicit object nodes, not just pixels.")