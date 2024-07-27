import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='Heart-counts.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)
    
    # Your code
    # part 2 and 3
#     silhouette = []
#     print("Silhouette coefficient:")
#     for i in range(2, 10):
#         kmeans = KMeans(n_clusters = i, init = 'random')
#         clustering = kmeans.fit(X)
#         sil_i = kmeans.silhouette(clustering, X)
#         silhouette.append(sil_i)
#         print("k = ", i, ": ", sil_i)
    
#     xticks = [2, 3, 4, 5, 6, 7, 8, 9]
#     plt.plot(xticks, silhouette)
#     plt.xlabel('number of clusters (k)')
#     plt.ylabel('silhouette score')
#     plt.show()
    
    # part 4
    kmeans = KMeans(n_clusters = 6, init = 'kmeans++')
    cluster = kmeans.fit(X)
    new_dim = PCA(X, 2)
    visualize_cluster(new_dim[:, 0], new_dim[:, 1], cluster)    

def visualize_cluster(x, y, clustering):
    labels = np.unique(clustering)
    for i in labels:
        x_new = x[clustering == i]
        y_new = y[clustering == i]
        plt.scatter(x_new, y_new)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
    
if __name__ == '__main__':
    main()
