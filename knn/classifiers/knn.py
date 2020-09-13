import numpy as np
class KNearestNeighbor():
    def __init__(self): 
        self.X_train = None 
        self.y_train = None 
        self.dists = None

    def train(self, X_train, y_train): 
        self.X_train = X_train 
        self.y_train = y_train 

    def compute_distance(self, X_test):
        num_train = self.X_train.shape[0]
        num_test = X_test.shape[0]
        dists = np.sum(self.X_train ** 2, 1).reshape(-1, num_train) - \
                2 * X_test.dot(self.X_train.T) + \
                np.sum(X_test ** 2, 1).reshape(num_test, -1)
        self.dists = dists 
        return dists

    def get_distances(self): 
        return self.dists
 
    def predict(self, X_test, k): 
        dists = self.compute_distance(X_test)
        num_test = X_test.shape[0]
        y_pred = np.zeros(num_test)
        dists_sorted = np.argsort(dists, axis=1)[:, :k]
        for i in range(num_test): 
            closest_y = self.y_train[dists_sorted[i,:]]
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred 
