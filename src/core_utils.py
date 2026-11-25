import os
import torch
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=4)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)

_cached_features = None
_cached_paths = None
_last_dir = None

def get_features(model, image_dir):
    global _cached_features, _cached_paths, _last_dir
    if _cached_features is not None and _last_dir == image_dir:
        print("Используются кэшированные признаки!")
        return _cached_features, _cached_paths

    print(f"Извлечение признаков из {image_dir}...")
    features, paths = [], []
    for cls in class_names:
        folder = os.path.join(image_dir, cls)
        if not os.path.exists(folder): continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, fname)
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    x = model.conv1(tensor)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)
                    x = model.avgpool(x)
                    features.append(torch.flatten(x, 1).cpu().numpy().squeeze())
                paths.append(path)
    
    _cached_features = np.vstack(features)
    _cached_paths = paths
    _last_dir = image_dir
    return _cached_features, _cached_paths

def visualize_clusters_2d(features, labels, title, use_tsne=False):
    if use_tsne:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
    else:
        reducer = PCA(n_components=2)
        xlabel, ylabel = "PCA Component 1", "PCA Component 2"
        
    pts = reducer.fit_transform(features)
    
    plt.figure(figsize=(9, 6))
    unique_labels = set(labels)
    cmap = plt.get_cmap('tab10', max(len(unique_labels), 10))
    
    for label in unique_labels:
        idxs = np.where(labels == label)
        color = 'k' if label == -1 else cmap(label)
        label_name = "Noise (-1)" if label == -1 else f"Cluster {label}"
        plt.scatter(pts[idxs, 0], pts[idxs, 1], label=label_name, s=30, color=color, edgecolors='white', linewidth=0.5)
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show(block=False)

def analyze_and_plot_samples(features, labels, paths):
    true_labels = [os.path.basename(os.path.dirname(p)) for p in paths]
    true_ids = [class_names.index(lbl) for lbl in true_labels]
    
    valid_mask = labels != -1
    
    # Расчет безразмерных метрик
    sil = silhouette_score(features[valid_mask], labels[valid_mask]) if len(set(labels[valid_mask])) > 1 else 0
    ari = adjusted_rand_score(true_ids, labels)
    nmi = normalized_mutual_info_score(true_ids, labels)
    
    # Визуализация нормализованных метрик
    fig_metrics, ax = plt.subplots(figsize=(8, 4))
    metrics_names = ['Silhouette\n(Плотность)', 'ARI\n(Совпадение с диагнозом)', 'NMI\n(Информативность)']
    scores = [sil, ari, nmi]
    
    bars = ax.bar(metrics_names, scores, color=['#3498db', '#2ecc71', '#9b59b6'])
    ax.set_ylim(-0.2, 1.0)
    ax.set_title("Нормализованные безразмерные оценки качества кластеризации", fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show(block=False)

    # Вывод примеров картинок
    unique_clusters = sorted(np.unique(labels))
    for cid in unique_clusters:
        if cid == -1: continue
        idxs = [i for i, lbl in enumerate(labels) if lbl == cid]
        assigned_true = [true_labels[i] for i in idxs]
        common, count = Counter(assigned_true).most_common(1)[0]
        purity = count / len(idxs)
        
        sample_idxs = [i for i in idxs if true_labels[i] == common][:5]
        if not sample_idxs: sample_idxs = idxs[:5]
        
        plt.figure(figsize=(12, 2.5))
        plt.suptitle(f"Примеры кластера {cid} (Доминирует: {common}, Чистота: {purity*100:.1f}%)", fontsize=10)
        for i, idx in enumerate(sample_idxs):
            img = Image.open(paths[idx]).convert('RGB')
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(os.path.basename(os.path.dirname(paths[idx])), fontsize=8)
            plt.axis('off')
        plt.show(block=False)
    
    plt.show() # Ожидаем закрытия всех окон