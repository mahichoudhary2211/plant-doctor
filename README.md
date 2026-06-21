Plant Doctor — AI-Powered Leaf Disease Diagnosis

An intelligent plant disease detection system combining **deep learning**, **explainable AI**, and **retrieval-augmented generation (RAG)** for automated diagnosis and treatment recommendations.

> Built as part of a team project at Bennett University.
> **My contributions: Grad-CAM explainability module + RAG-based treatment recommendation system**

---

How It Works

1. User uploads a leaf image
2. **MobileNetV2** classifies the disease
3. **Grad-CAM** generates a heatmap highlighting *which part of the leaf* the model focused on
4. **RAG (TF-IDF)** retrieves verified treatment recommendations from a curated knowledge base
5. Results displayed on a responsive web interface

---

My Contributions

Grad-CAM Explainability (`grad_cam.py`)
- Implemented Gradient-weighted Class Activation Mapping
- Generates visual heatmaps overlaid on leaf images
- Shows *exactly* which regions the model used to make its prediction
- Builds user trust by making the AI decision transparent

RAG Treatment Recommendation (`rag.py`)
- Built TF-IDF vectorization pipeline over a curated disease knowledge base
- Cosine similarity retrieval ensures factually accurate treatment suggestions
- Zero hallucinations — all responses grounded in verified agricultural sources
- Achieves cosine similarity scores >0.7 for matched disease-treatment pairs

---

Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.0, MobileNetV2 |
| Explainability | Grad-CAM (custom implementation) |
| RAG | TF-IDF + Cosine Similarity (scikit-learn) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Dataset | PlantVillage |

---

Results

- Inference speed: 50–100ms on CPU, <20ms on GPU
- Model size: ~14 MB
- RAG retrieval accuracy: cosine similarity >0.7
- Grad-CAM successfully localizes lesions, chlorotic areas, and necrotic tissue

---

Run Locally

```bash
git clone https://github.com/mahichoudhary2211/plant-doctor
cd plant-doctor
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

---

Contributors

- [Shreyansh Tiwari](https://github.com/Conspirer) — MobileNetV2 classification, FastAPI backend
- [Mahi Choudhary](https://github.com/mahichoudhary2211) — Grad-CAM explainability, RAG system

---

## 📄 Reference

Based on research submitted to Bennett University, Dept. of CSE.
