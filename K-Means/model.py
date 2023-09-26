import numpy as np
import random as random
import math as math
from numpy import linalg as LA
from tqdm import trange, tqdm
from utils import error

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon

    def euclid(self, a, b):
        return LA.norm(a-b)

    def classify(self, p, means):
        closest_means = -1
        min_dist = math.inf
        for i in range(0, len(means)):
            temp = self.euclid(p, means[i])
            if(temp < min_dist):
                closest_means = i
                min_dist = temp
        return closest_means

    def initialise_means(self, points, k):
        means = []
        index = [random.sample(range(0, len(points)), k)]
        for i in index:
            means.append(points[i])

        return means[0]

    def recompute_means(self, A, points, k):
        means = []
        for i in range(0, k):
            length = 0
            kth_mean = np.zeros(len(points[0]))
            for j in range(0, len(points)):
                if(A[i][j] == 1):
                    kth_mean += points[j]
                    length += 1
            length = 1 if(length == 0) else length
            kth_mean = kth_mean*(1/length)

            means.append(kth_mean)
        return means

    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        means = []
        index = [random.sample(range(0, len(X)), self.num_clusters)]
        for i in index:
            means.append(X[i])
        means = means[0]
        
        for i in tqdm(range(max_iter)):
            # Assign each sample to the closest prototype
            new_means = []
            A = [[0]*len(X) for i in range(self.num_clusters)]
            for j in range(len(X)):
                
                index = self.classify(X[j],means)

                A[index][j] = 1

            new_means = self.recompute_means(A, X, self.num_clusters)

            if (np.array_equal(means, new_means) or i==(max_iter-1) or error(np.array(means),np.array(new_means))<=self.epsilon):
                self.cluster_centers = new_means
                break
            means = new_means
            

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        mean_indexes = []
        for i in range(len(X)):
            mean_indexes.append(self.classify(X[i], self.cluster_centers))

        return mean_indexes

    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)

    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        clustered_image = []
        for i in range(len(X)):
            clustered_image.append(
                self.cluster_centers[self.classify(X[i], self.cluster_centers)])

        return np.array(clustered_image)
