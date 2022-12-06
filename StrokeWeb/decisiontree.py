import numpy as np
from collections import Counter
def entropy(y):
    hist = np.bincount(y) # liệt kê các phần tử và số lần xuất hiện của nó
    ps = hist / len(y) # tỉ lệ của các phần tử
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.X_train = None
        self.y_train = None
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        # phat trien cay
        self.root = self.grow(X, y)

    def predict(self, X):
        return np.array([self.move_in_tree(x, self.root) for x in X]) 

    def grow(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # xac dinh so nhan co trong mang
        n_labels = len(np.unique(y))
        # điều kiện dừng khi đạt đến độ sâu tối đa hoặc chỉ có 1 nhãn
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # tạo 1 sắp xếp ngẫu nhiêu các feature
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # chọn đường đi tốt nhất
        best_feat, best_thresh = self.best_road(X, y, feat_idxs)
    
        # phát triển tiếp cây quyết định từ nút cha là các mảng vừa mới được chia
        left_idxs, right_idxs = self.split(X[:, best_feat], best_thresh)
        left = self.grow(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.grow(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def best_road(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def information_gain(self, y, X_column, split_thresh):
        # tính entropy tại nút cha
        parent_entropy = entropy(y)

        # chia thành các lá
        left_idxs, right_idxs = self.split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Tính entropy của nút con
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # Tính gain
        ig = parent_entropy - child_entropy
        return ig

    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() # Lấy mảng chứa vị trí các phần tử có giá trị nhỏ hơn giá trị split_thresh và chuyển về dạng mảng 1 chiều
        right_idxs = np.argwhere(X_column >= split_thresh).flatten() # Lấy mảng chứa vị trí các phần tử có giá trị lớn hơn giá trị split_thresh
        return left_idxs, right_idxs

    def move_in_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold: # nếu giá trị nhỏ hơn ngưỡng thì đi bên trái
            return self.move_in_tree(x, node.left)
        return self.move_in_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        if(len(counter)!=0):
            most_common = counter.most_common(1)[0][0]
            return most_common
        else:
            return [0]
