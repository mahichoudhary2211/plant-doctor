import io, base64, os, glob, requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import models
import torch
from utils import load_image_as_tensor
from grad_cam import GradCAM, overlay_heatmap
from rag import SimpleRAG
import cv2

APP = FastAPI(title="Plant Doctor")

# ---------- Static & Home ----------
APP.mount("/static", StaticFiles(directory="static"), name="static")

@APP.get("/")
def root():
    return FileResponse("static/index.html")

@APP.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")

# ---------- Model download ----------
MODEL_PATH = "models/plant_doctor.pt"
MODEL_URL = "https://drive.google.com/uc?id=1AQMLO8OMVYs4fy7KqCCnjUQcbrMciX-R"

model = None
classes = None

def load_model():
    global model, classes

    if model is not None:
        return model, classes

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    classes = ckpt["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, classes

# ---------- RAG ----------
RAG = SimpleRAG(["rag_kb/treatments.md"])

# ---------- Health ----------
@APP.get("/health")
def health():
    m, c = load_model()
    return {"status": "ok", "num_classes": len(c)}

# ---------- Stats ----------
def _count_images_by_class(data_root="data/plantvillage"):
    counts = {}
    if not os.path.isdir(data_root):
        return counts
    for cls in os.listdir(data_root):
        p = os.path.join(data_root, cls)
        if os.path.isdir(p):
            n = len(glob.glob(os.path.join(p, "*")))
            counts[cls] = n
    return counts

@APP.get("/stats")
def stats():
    _, c = load_model()
    img_counts = _count_images_by_class()
    return {
        "model": "plant_doctor.pt",
        "classes": c,
        "num_classes": len(c),
        "image_counts": img_counts,
        "total_images": sum(img_counts.values()) if img_counts else 0,
    }

# ---------- Predict ----------
@APP.post("/predict")
async def predict(file: UploadFile = File(...)):
    model, classes = load_model()

    raw = await file.read()
    x = load_image_as_tensor(io.BytesIO(raw))

    cam_gen = GradCAM(model)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)
    pred = classes[idx.item()]

    k = min(3, len(classes))
    topk_vals, topk_idxs = torch.topk(probs, k=k)
    top3 = [
        {"label": classes[i.item()], "prob": float(p)}
        for i, p in zip(topk_idxs[0], topk_vals[0])
    ]

    heatmap_b64 = ""
    try:
        cam = cam_gen(logits, idx.item())
        img = tensor_to_rgb(x[0])
        overlay = overlay_heatmap(img[:, :, ::-1], cam)
        _, buf = cv2.imencode(".png", overlay)
        heatmap_b64 = base64.b64encode(buf).decode()
    except:
        pass

    query = f"{pred.replace('_', ' ')} disease symptoms treatment control prevention"
    hits = RAG.retrieve(query, topk=3)

    exact = RAG.get_by_title(pred)
    if exact:
        treatment = exact
        source = pred
    elif hits:
        source, treatment, _ = hits[0]
    else:
        treatment = "No treatment found."
        source = ""

    return JSONResponse({
        "prediction": pred,
        "confidence": float(conf.detach()),
        "top3": top3,
        "heatmap_image_b64": heatmap_b64,
        "treatment_plan": treatment,
        "source": source
    })

# ---------- Chat ----------
@APP.post("/chat")
async def chat(query: str):
    hits = RAG.retrieve(query)
    answer = hits[0][1] if hits else "No info found."
    source = hits[0][0] if hits else ""
    return {"query": query, "answer": answer, "source": source}

# ---------- Helper ----------
def tensor_to_rgb(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = t.permute(1,2,0).detach().cpu().numpy()
    img = (img * std + mean)
    img = (img * 255).clip(0,255).astype("uint8")
    return img
