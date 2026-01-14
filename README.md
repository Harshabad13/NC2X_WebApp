---
title: NC2X Concept Causal AI
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
python_version: "3.10"
license: mit
short_description: A Neuro-Symbolic Framework for Causal Scene Understanding
---

# 🧠 NC2X: Concept, Causal & Context-Aware AI

### A Neuro-Symbolic Framework for Explainable Scene Understanding

**NC2X** is an advanced AI system that bridges the gap between **Perception** (Seeing) and **Reasoning** (Understanding). Unlike traditional "Black Box" AI models that only detect objects, NC2X understands the **Context** and **Causal Relationships** between them.

---

## 🚀 Key Features

1.  **👁️ Object Detection (Perception):**
    *   Uses **YOLOv8** to accurately detect objects in real-time.
    *   Identifies 80+ types of objects (Person, Chair, Laptop, etc.).

2.  **🕸️ Contextual Reasoning (The Brain):**
    *   Constructs a **Scene Graph** connecting detected objects.
    *   Uses a custom **Graph Neural Network (GNN)** to understand relationships (e.g., *Dining Table is defined by Chairs around it*).

3.  **🧪 Causal Analysis (The "Why" Factor):**
    *   Performs **Visual Masking Interventions**.
    *   You can remove an object (e.g., Chair) and see how the model's confidence for related objects (e.g., Table) drops.
    *   This proves the model is **reasoning**, not just memorizing.

4.  **🆚 Comparative Analysis:**
    *   Compare **NC2X** (Concept-based) vs **Grad-CAM** (Pixel-based) explanations side-by-side.

---

## 🛠️ Tech Stack

*   **Framework:** PyTorch & PyTorch Geometric
*   **Models:** YOLOv8n + ResNet-50 + Custom GCN
*   **Interface:** Streamlit
*   **Visualization:** OpenCV & Streamlit-Agraph

---

## 📖 How to Use

1.  **Image Analysis:** Upload an image to see the Scene Graph and Top Concepts.
2.  **Video Analysis:** Upload a video to see real-time context tracking.
3.  **Live Webcam:** Use your camera for real-time scene
