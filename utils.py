import torch
from torchvision import transforms
from PIL import Image

IMG_SIZE = 224

inference_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image_as_tensor(fp):
    img = Image.open(fp).convert('RGB')
    return inference_tfms(img).unsqueeze(0)
