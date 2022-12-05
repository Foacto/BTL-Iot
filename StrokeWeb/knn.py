from collections import Counter

import numpy as np

def euclidean_distance(x1, x2):
    # TÌm khoảng cách giữa 2 mẫu
    return np.sqrt(np.sum((x1 - x2) ** 2))

class Custom_KNN:
    def __init__(self, k):
        self.counterX = 0
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        if (self.counterX % 1000) == 0:
            print(self.counterX)
        self.counterX += 1
        # Tính khoảng cách giữa x và tất cả các mẫu trong bộ dữ liệu
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # sắp xếp và lấy k mẫu có khoảng cách gần với mẫu nhập nhất
        k_idx = np.argsort(distances)[: self.k]
        # Lấy tập các nhãn của k mẫu đó
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # print(k_neighbor_labels)
        # trả về nhãn có số lần xuất hiện nhiều nhất
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
