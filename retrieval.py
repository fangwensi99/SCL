
import os

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm
from datetime import datetime


class Config:
    DATASET = "officehome"
    DOMAINS = ["Art", "Clipart", "Product", "Real_World"]

    # Retrieval parameters
    # PRECISION_K = [1, 5, 15]  # Precision@K metrics
    PRECISION_K = [50, 100, 200]
    BATCH_SIZE = 1024
    RETRIEVAL_BATCH_SIZE = 512  # Batch size for retrieval computation to avoid OOM

    # Output configuration
    OUTPUT_DIR = "./results/"
    SAVE_RESULTS = True
    SAVE_LOG = True  # Save log to txt file

def compute_similarity_matrix(features1, features2, metric="cosine", batch_size=None):
    """
    Compute similarity matrix between two feature sets using GPU with batching to avoid OOM
    Returns numpy array to save memory
    """
    print(f"  Computing similarity matrix: {features1.shape} x {features2.shape}")

    if batch_size is None:
        # Auto-determine batch size based on available memory
        # Estimate: each similarity value is 4 bytes (float32)
        # For safety, use smaller batches
        batch_size = min(512, features1.shape[0])

    # Normalize gallery features once (features2)
    feat2_tensor = torch.from_numpy(features2).float().cuda()
    if metric == "cosine":
        feat2_tensor = F.normalize(feat2_tensor, p=2, dim=1)

    # Compute similarity in batches
    similarity_list = []
    feat1_tensor = torch.from_numpy(features1).float().cuda()

    for i in range(0, features1.shape[0], batch_size):
        end_idx = min(i + batch_size, features1.shape[0])
        feat1_batch = feat1_tensor[i:end_idx]

        if metric == "cosine":
            feat1_batch = F.normalize(feat1_batch, p=2, dim=1)

        # Compute similarity for this batch
        batch_sim = torch.matmul(feat1_batch, feat2_tensor.T)  # (batch_size, N2)
        similarity_list.append(batch_sim.cpu().numpy())

        # Clear GPU cache
        del batch_sim
        torch.cuda.empty_cache()

    # Concatenate all batches
    similarity = np.concatenate(similarity_list, axis=0)

    # Clear tensors
    del feat1_tensor, feat2_tensor
    torch.cuda.empty_cache()

    return similarity

def retrieval_precision_cal(features_A, targets_A, features_B, targets_B, preck=(1, 5, 15), batch_size=512):
    """
    Calculate precision@k metrics for bidirectional retrieval using GPU with batching

    Args:
        features_A: Features of domain A (N_A, D) - numpy array
        targets_A: Labels of domain A (N_A,)
        features_B: Features of domain B (N_B, D) - numpy array
        targets_B: Labels of domain B (N_B,)
        preck: List of k values for precision@k
        batch_size: Batch size for processing queries to avoid OOM

    Returns:
        res_A: precision@k results for domain A as query
        res_B: precision@k results for domain B as query
    """
    # Normalize gallery features once (features_B)
    feat_B_tensor = torch.from_numpy(features_B).float().cuda()
    feat_B_tensor = F.normalize(feat_B_tensor, p=2, dim=1)
    targets_B_tensor = torch.from_numpy(targets_B).long().cuda()

    # Normalize query features (features_A)
    feat_A_tensor = torch.from_numpy(features_A).float().cuda()
    feat_A_tensor = F.normalize(feat_A_tensor, p=2, dim=1)
    targets_A_tensor = torch.from_numpy(targets_A).long().cuda()

    res_A = []
    res_B = []

    # Process A -> B direction
    print(f"    Computing A->B precision (batch size: {batch_size})...")
    precision_A = _compute_precision_batched(
        feat_A_tensor, targets_A_tensor,
        feat_B_tensor, targets_B_tensor,
        preck, batch_size
    )
    res_A = precision_A

    # Process B -> A direction
    print(f"    Computing B->A precision (batch size: {batch_size})...")
    precision_B = _compute_precision_batched(
        feat_B_tensor, targets_B_tensor,
        feat_A_tensor, targets_A_tensor,
        preck, batch_size
    )
    res_B = precision_B

    # Clear GPU cache
    del feat_A_tensor, feat_B_tensor, targets_A_tensor, targets_B_tensor
    torch.cuda.empty_cache()

    return res_A, res_B


def cross_domain_retrieval(source_domain, target_domain, config):
    """Perform cross-domain retrieval"""
    print(f"\n🔍 Cross-domain retrieval: {source_domain} ↔ {target_domain}")

    # Load source and target domain data
    source_img, source_text, source_labels = load_domain_features(source_domain, config.DATASET, config)
    target_img, target_text, target_labels = load_domain_features(target_domain, config.DATASET, config)

    # Combine features
    print(f"  Text feature status: source={'exists' if source_text is not None else 'missing'}, target={'exists' if target_text is not None else 'missing'}")
    print(f"  Retrieval mode: {Config.RETRIEVAL_METHOD}")

    source_features = load_features(source_img, source_text, config.RETRIEVAL_METHOD, config.CONCAT_WEIGHT)
    target_features = load_features(target_img, target_text, config.RETRIEVAL_METHOD, config.CONCAT_WEIGHT)
    print(f"  Combined features: source={source_features.shape}, target={target_features.shape}")

    # Compute similarity matrix (returns numpy array, computed in batches)
    similarity_matrix = compute_similarity_matrix(
        source_features, target_features,
        batch_size=config.RETRIEVAL_BATCH_SIZE
    )

    # Calculate precision@k metrics for bidirectional retrieval
    retrieval_metrics = {}
    if source_labels is not None and target_labels is not None:
        print(f"  Computing precision@k metrics for bidirectional retrieval...")

        # Use precision calculation function with batching
        precision_A, precision_B = retrieval_precision_cal(
            source_features, source_labels,
            target_features, target_labels,
            config.PRECISION_K,
            batch_size=config.RETRIEVAL_BATCH_SIZE
        )

        # Store results
        for i, k in enumerate(config.PRECISION_K):
            retrieval_metrics[f'{source_domain}_to_{target_domain}_P@{k}'] = precision_A[i]
            retrieval_metrics[f'{target_domain}_to_{source_domain}_P@{k}'] = precision_B[i]

        # Calculate Precision@All - use all relevant sample counts
        # For each query, calculate the number of same-class samples in gallery, take maximum as all
        max_relevant_A = max([(target_labels == label).sum() for label in np.unique(source_labels)])
        max_relevant_B = max([(source_labels == label).sum() for label in np.unique(target_labels)])
        max_all = max(max_relevant_A, max_relevant_B)

        precision_all_A, precision_all_B = retrieval_precision_cal(
            source_features, source_labels,
            target_features, target_labels,
            [max_all]  # Use all relevant sample counts as all
        )

        retrieval_metrics[f'{source_domain}_to_{target_domain}_P@all'] = precision_all_A[0]
        retrieval_metrics[f'{target_domain}_to_{source_domain}_P@all'] = precision_all_B[0]

        # Print results
        print(f"  {source_domain} → {target_domain}:")
        for i, k in enumerate(config.PRECISION_K):
            print(f"    P@{k}: {precision_A[i]:.2f}%")
        print(f"    P@all: {precision_all_A[0]:.2f}%")

        print(f"  {target_domain} → {source_domain}:")
        for i, k in enumerate(config.PRECISION_K):
            print(f"    P@{k}: {precision_B[i]:.2f}%")
        print(f"    P@all: {precision_all_B[0]:.2f}%")
    else:
        print("  ⚠️  No label information, cannot calculate precision metrics")

    return {
        'similarity_matrix': similarity_matrix,
        'metrics': retrieval_metrics,
        'source_features': source_features,
        'target_features': target_features,
        'source_labels': source_labels,
        'target_labels': target_labels
    }

def evaluate_clustering(features, labels, n_clusters):
    """Evaluate clustering performance"""
    print(f"  Executing K-means clustering (k={n_clusters})...")

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    # Calculate clustering metrics
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)

    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")

    return {
        'cluster_labels': cluster_labels,
        'nmi': nmi,
        'ari': ari
    }

def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    log_content = []
    if config.SAVE_LOG:
        from datetime import datetime
        log_file = os.path.join(config.OUTPUT_DIR, f"retrieval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        # Save original print function
        import builtins
        original_print = builtins.print

        def log_print(*args, **kwargs):
            message = ' '.join(map(str, args))
            original_print(*args, **kwargs)  # Use original print function
            log_content.append(message)

        builtins.print = log_print
    else:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)

    available_domains = []
    for domain in config.DOMAINS:
        img_path = config.IMAGE_FEAT_TEMPLATE.format(dataset=config.DATASET, domain=domain)
        if os.path.exists(img_path):
            available_domains.append(domain)

    all_results = {}
    domain_pairs = [(d1, d2) for i, d1 in enumerate(available_domains)
                   for d2 in available_domains[i+1:]]

    for i, (source_domain, target_domain) in enumerate(domain_pairs, 1):
        print(f"\n[{i}/{len(domain_pairs)}] Processing: {source_domain} → {target_domain}")

        try:
            # Execute retrieval
            result = cross_domain_retrieval(source_domain, target_domain, config)
            all_results[f"{source_domain}_to_{target_domain}"] = result

            # If labels exist, evaluate clustering performance
            if result['source_labels'] is not None:
                n_clusters = len(np.unique(result['source_labels']))
                clustering_result = evaluate_clustering(
                    result['source_features'],
                    result['source_labels'],
                    n_clusters
                )
                result['clustering'] = clustering_result

    # Save log
    if config.SAVE_LOG:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))
        original_print(f" Log saved: {log_file}")

        # Restore original print function
        import builtins
        builtins.print = original_print

    # Output summary
    print(f"Retrieval Results Summary:")
    for key, result in all_results.items():
        if 'metrics' in result and result['metrics']:
            print(f"\n{key}:")

            # Display precision metrics grouped by domain pairs
            source_domain, target_domain = key.split('_to_')

            print(f"  {source_domain} → {target_domain}:")
            for k in config.PRECISION_K:
                metric_key = f'{source_domain}_to_{target_domain}_P@{k}'
                if metric_key in result['metrics']:
                    print(f"    P@{k}: {result['metrics'][metric_key]:.2f}%")

            all_key = f'{source_domain}_to_{target_domain}_P@all'
            if all_key in result['metrics']:
                print(f"    P@all: {result['metrics'][all_key]:.2f}%")

            print(f"  {target_domain} → {source_domain}:")
            for k in config.PRECISION_K:
                metric_key = f'{target_domain}_to_{source_domain}_P@{k}'
                if metric_key in result['metrics']:
                    print(f"    P@{k}: {result['metrics'][metric_key]:.2f}%")

            all_key = f'{target_domain}_to_{source_domain}_P@all'
            if all_key in result['metrics']:
                print(f"    P@all: {result['metrics'][all_key]:.2f}%")

            if 'clustering' in result:
                print(f"  Clustering metrics:")
                print(f"    NMI: {result['clustering']['nmi']:.4f}")
                print(f"    ARI: {result['clustering']['ari']:.4f}")

if __name__ == "__main__":
    main()
