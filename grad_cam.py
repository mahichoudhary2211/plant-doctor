import torch, cv2, numpy as np

class GradCAM:
    def __init__(self, model, target_layer_name='features.18.0'):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        target_layer = dict([*model.named_modules()])[target_layer_name]
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, scores, class_idx):
        self.model.zero_grad()
        scores[0, class_idx].backward(retain_graph=True)
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights*acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min())/(cam.max() + 1e-8)
        return cam

def overlay_heatmap(img_rgb, cam):
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.4*heatmap + 0.6*img_rgb).astype(np.uint8)
    return overlay
