import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from core_utils import class_names

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        self.target_layer = model.layer4[-1]
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)
        
    def save_feature_maps(self, module, input, output): self.feature_maps = output
    def save_gradients(self, module, grad_input, grad_output): self.gradients = grad_output[0]
        
    def generate_heatmap(self, image_tensor, class_idx):
        outputs = self.model(image_tensor)
        score = outputs[0][class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        gradients = self.gradients.detach().cpu().numpy()[0]
        feature_maps = self.feature_maps.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights): heatmap += w * feature_maps[i]
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0: heatmap = heatmap / np.max(heatmap)
        return heatmap

def predict_and_visualize(image_path, model, grad_cam):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_pil = Image.open(image_path).convert('RGB')
    img_display_pil = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(img_pil)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = transform(img_pil).unsqueeze(0).to(device)
    model.eval()
    
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]
    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_idx]
    
    heatmap = grad_cam.generate_heatmap(image_tensor, predicted_idx)
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_pil = transforms.Resize((224, 224))(heatmap_pil)
    heatmap_np = np.array(heatmap_pil) / 255.0
    
    cm = plt.get_cmap('jet')
    heatmap_color = cm(heatmap_np)[:, :, :3]
    orig_np = np.array(img_display_pil) / 255.0
    alpha = 0.45
    blended_image = orig_np * (1.0 - heatmap_np[:, :, None] * alpha) + heatmap_color * (heatmap_np[:, :, None] * alpha)
    return predicted_class, blended_image, orig_np, probabilities.detach().cpu().numpy() * 100

def run_gradcam_single(model, image_path, true_class):
    grad_cam = GradCAM(model)
    predicted_class, blended_img, orig_img, probs = predict_and_visualize(image_path, model, grad_cam)

    fig, (ax_orig, ax_grad, ax_bar) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.15, wspace=0.3)
    
    ax_orig.imshow(orig_img)
    ax_orig.set_title("Оригинальный снимок МРТ", fontsize=11, fontweight='bold')
    ax_orig.axis('off')
    
    ax_grad.imshow(blended_img)
    title_grad = ax_grad.set_title(f"ИИ: {predicted_class}\nФакт: {true_class}", fontsize=11, fontweight='bold')
    title_grad.set_color('green' if predicted_class == true_class else 'red')
    ax_grad.axis('off')
    
    bars = ax_bar.bar(class_names, probs, color=['#ff4d4d', '#33cc33', '#3399ff', '#cc33ff'])
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel('Уверенность ИИ (%)', fontsize=10)
    ax_bar.set_title("Анализ вероятностей", fontsize=11, fontweight='bold')
    ax_bar.tick_params(axis='x', rotation=15)
    
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.show()