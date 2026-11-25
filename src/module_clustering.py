import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from core_utils import visualize_clusters_2d, analyze_and_plot_samples

def run_kmeans(features, paths):
    print("\nЗапуск K-Means...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(features)
    visualize_clusters_2d(features, labels, "K-Means Clustering (PCA)", use_tsne=False)
    analyze_and_plot_samples(features, labels, paths)

def run_agglomerative(features, paths):
    print("\nЗапуск Agglomerative Clustering...")
    agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels = agg.fit_predict(features)
    visualize_clusters_2d(features, labels, "Agglomerative Clustering (PCA)", use_tsne=False)
    analyze_and_plot_samples(features, labels, paths)

def run_gmm(features, paths):
    print("\nЗапуск Gaussian Mixture Models...")
    pca_reduction = PCA(n_components=50, random_state=42)
    feats_reduced = pca_reduction.fit_transform(features)
    gmm = GaussianMixture(n_components=4, covariance_type='full', reg_covar=1e-3, random_state=42)
    labels = gmm.fit_predict(feats_reduced)
    visualize_clusters_2d(features, labels, "GMM Clustering (PCA)", use_tsne=False)
    analyze_and_plot_samples(features, labels, paths)

def run_dbscan_auto(features, paths):
    """Вариант 1: DBSCAN сам определяет количество кластеров."""
    print("\nЗапуск DBSCAN (Естественное количество кластеров)...")
    pca_high = PCA(n_components=30, random_state=42)
    features_30d = pca_high.fit_transform(features)
    
    # Автоматический расчет eps
    min_samples = 5
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(features_30d)
    distances, _ = nn.kneighbors(features_30d)
    auto_eps = np.mean(distances[:, -1]) * 1.5
    
    print(f"Автоматический радиус (eps) = {auto_eps:.2f}")

    dbscan = DBSCAN(eps=auto_eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_30d)
    
    actual_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Алгоритм выделил естественных кластеров: {actual_clusters}")
    
    visualize_clusters_2d(features_30d, labels, f"DBSCAN Auto ({actual_clusters} cls, eps={auto_eps:.2f})", use_tsne=True)
    analyze_and_plot_samples(features, labels, paths)

def run_dbscan_forced(features, paths, target_clusters=4):
    """Вариант 2: Заставляем DBSCAN найти заданное количество кластеров."""
    print(f"\nЗапуск DBSCAN (Принудительный поиск {target_clusters} кластеров)...")
    pca_high = PCA(n_components=30, random_state=42)
    features_30d = pca_high.fit_transform(features)

    # Автоматический расчет eps оставляем как базу
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(features_30d)
    distances, _ = nn.kneighbors(features_30d)
    auto_eps = np.mean(distances[:, -1]) * 1.5
    
    print(f"Фиксированный автоматический радиус (eps) = {auto_eps:.2f}")
    
    best_labels = None
    closest_diff = float('inf')

    # Цикл-обертка: меняем плотность (min_samples), чтобы "заставить" алгоритм выдать n кластеров
    for min_s in range(2, 60):
        dbscan = DBSCAN(eps=auto_eps, min_samples=min_s)
        labels = dbscan.fit_predict(features_30d)
        
        # Считаем количество кластеров (игнорируя шум -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Если нашли идеальное совпадение
        if n_clusters == target_clusters:
            best_labels = labels
            break
        
        # Запоминаем наиболее близкий вариант, если идеального не будет
        if abs(n_clusters - target_clusters) < closest_diff:
            closest_diff = abs(n_clusters - target_clusters)
            best_labels = labels

    actual_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    
    if actual_clusters == target_clusters:
        print(f"Успешно: данные разбиты ровно на {actual_clusters} кластера.")
    else:
        print(f"Внимание: топология данных не позволяет выделить ровно {target_clusters} кластера при текущем eps. Найдено наиболее близкое число: {actual_clusters}.")

    visualize_clusters_2d(features_30d, best_labels, f"DBSCAN Forced ({actual_clusters} cls, eps={auto_eps:.2f})", use_tsne=True)
    analyze_and_plot_samples(features, best_labels, paths)