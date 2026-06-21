import torch, base64, cv2
from torchvision import models
from utils import load_image_as_tensor
from grad_cam import GradCAM, overlay_heatmap

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(ckpt['classes']))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['classes']

def predict_with_cam(image_path, ckpt_path):
    model, classes = load_model(ckpt_path)
    x = load_image_as_tensor(image_path)

    # Init Grad-CAM BEFORE forward so hooks capture activations
    cam_gen = GradCAM(model)

    # IMPORTANT: do NOT use torch.no_grad() here
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)

    # Grad-CAM
    cam = cam_gen(logits, idx.item())

    # make overlay image
    img = (x[0].permute(1,2,0).numpy()*[0.229,0.224,0.225] + [0.485,0.456,0.406])
    img = (img*255).clip(0,255).astype('uint8')
    img_bgr = img[:, :, ::-1]
    overlay = overlay_heatmap(img_bgr, cam)
    _, buf = cv2.imencode('.png', overlay)
    b64 = base64.b64encode(buf).decode()
    return classes[idx], float(conf), b64


if __name__ == '__main__':
    import sys
    label, conf, b64 = predict_with_cam(sys.argv[1], 'models/plant_doctor.pt')
    print(label, conf)
