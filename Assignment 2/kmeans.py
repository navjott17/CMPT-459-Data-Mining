import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """
        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def make_clusters(self, X: np.ndarray):
        distance = self.euclidean_distance(X, self.centroids)
        minimum_indexes = [np.where(distance[i] == np.min(distance[i])) for i in range(distance.shape[0])]
        minimum_indexes = np.array(minimum_indexes)
        minimum_indexes = minimum_indexes.reshape(minimum_indexes.shape[0])
        return minimum_indexes 

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        flag = 0
        min_idx = self.make_clusters(X)
        while iteration < self.max_iter:
            update_idx = np.zeros(X.shape[0])
            if (flag):
                break
            else:
                self.update_centroids(min_idx, X)
                update_idx = self.make_clusters(X)
            flag = np.array(min_idx == update_idx)
            flag = 1 if all(flag) else 0
            min_idx = update_idx
            clustering = min_idx
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        updation_of_centroids = []
        for i in range(self.n_clusters):
            cluster = [X[j] for j in range(X.shape[0]) if clustering[j] == i]
            cluster = np.array(cluster)
            mean = np.mean(cluster, axis = 0)
            updation_of_centroids.append(mean)
        self.centroids = np.array(updation_of_centroids)
        
    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        row = X.shape[0]
        if self.init == 'random':
            idx = np.random.choice(row, self.n_clusters, replace = False)
            self.centroids = np.array([X[idx[i]] for i in range(self.n_clusters)])
            
        elif self.init == 'kmeans++':
            centroids = []
            index = np.random.choice(row, replace = False)
            centroids.append(X[index])
            for i in range(self.n_clusters-1):
                distance = np.square(self.euclidean_distance(X, np.array(centroids)))
                min_distance = np.min(distance, axis = 1)
                probabilities = min_distance/np.sum(min_distance)
                probabilities = probabilities.flatten()
                idx = np.random.choice(row, replace = False, p = probabilities)
                centroids.append(X[idx])
            self.centroids = np.array(centroids)
            
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        row_x1 = X1.shape[0]
        col_x1 = X1.shape[1]
        row_x2 = X2.shape[0]
        col_x2 = X2.shape[1]
        if(row_x1 > row_x2):
            size_max = row_x1
            size_min = row_x2
            large = X1
            small = X2
        else:
            size_max = row_x2
            size_min = row_x1
            large = X2
            small = X1

        distance = [np.sqrt(np.sum(np.square(large[i] - small[j]))) for i in range(size_max) for j in range(size_min)]
        distance = np.array(distance).reshape(large.shape[0], small.shape[0])
        return distance

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        sil = []
        for i in range(X.shape[0]):
            data = X[i].reshape(1, X.shape[1])
            clus_data = clustering[i]
            distance_own = []
            distance_diff = []
            for j in range(self.n_clusters):
                if j == clus_data:
                    distance_own.append(self.euclidean_distance(data, np.array(self.centroids[j]).reshape(1, X.shape[1])))
                else:
                    distance_diff.append(self.euclidean_distance(data, np.array(self.centroids[j]).reshape(1, X.shape[1])))
            a_o = np.min(distance_own)
            b_o = np.min(distance_diff)
            s_o = (b_o - a_o)/np.maximum(a_o, b_o)
            sil.append(s_o)
        sil = np.mean(sil)
        return sil
