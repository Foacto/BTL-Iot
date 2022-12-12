from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np


# def getQttPerLabel(y):
#     trueQtt = 0
#     for i in y:
#         if y[i] == 1:
#             trueQtt += 1
#     return np.array([len(y) - trueQtt, trueQtt])


def tinhE(y):
    # y = [0, 1, 0] --> soLuongCacNhan = [2, 1]
    soLuongCacNhan = np.bincount(y)
    # soLuongCacNhan = getQttPerLabel(y)
    cacXacSuat = soLuongCacNhan / len(y)
    return -np.sum([xacSuat * np.log2(xacSuat) for xacSuat in cacXacSuat if xacSuat > 0])


class Node:
    def __init__(
        self, feature=None, nguong=None, conBenTrai=None, conBenPhai=None, *, value=None
    ):
        self.feature = feature
        self.nguong = nguong
        self.conBenTrai = conBenTrai
        self.conBenPhai = conBenPhai
        self.value = value

    def laNodeLa(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, doSauToiDa=100, soLuongFeatures=None):
        self.min_samples_split = min_samples_split
        self.doSauToiDa = doSauToiDa
        self.soLuongFeatures = soLuongFeatures
        self.nodeGoc = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.soLuongFeatures = X.shape[1] if not self.soLuongFeatures else min(
            self.soLuongFeatures, X.shape[1])
        self.nodeGoc = self.phatTrienCay(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.nodeGoc) for x in X])

    def phatTrienCay(self, X, y, doSauHienTai=0):
        soLuongMau, soLuongFeatures = X.shape
        soLuongNhan = len(np.unique(y))

        if (
            doSauHienTai >= self.doSauToiDa
            or soLuongNhan == 1
            or soLuongMau < self.min_samples_split
        ):
            leaf_value = self.getLabel(y)
            return Node(value=leaf_value)

        featureIndexs = np.random.choice(
            soLuongFeatures, self.soLuongFeatures, replace=False)

        best_feat_index, best_nguong = self.getBestFeatIdxAndSplitThresh(
            X, y, featureIndexs)

        conBenTraiIdxs, conBenPhaiIdxs = self._split(
            X[:, best_feat_index], best_nguong)
        conBenTrai = self.phatTrienCay(
            X[conBenTraiIdxs, :], y[conBenTraiIdxs], doSauHienTai + 1)
        conBenPhai = self.phatTrienCay(
            X[conBenPhaiIdxs, :], y[conBenPhaiIdxs], doSauHienTai + 1)

        return Node(best_feat_index, best_nguong, conBenTrai, conBenPhai)

    def getBestFeatIdxAndSplitThresh(self, X, y, featureIndexs):
        doLoiThongTinCaoNhat = -1
        split_idx, split_thresh = None, None

        for feat_idx in featureIndexs:
            cotHienTai = X[:, feat_idx]
            nguongs = np.unique(cotHienTai)

            for nguong in nguongs:
                doLoiThongTin = self.getInformationGain(y, cotHienTai, nguong)

                if doLoiThongTin > doLoiThongTinCaoNhat:
                    doLoiThongTinCaoNhat = doLoiThongTin
                    split_idx = feat_idx
                    split_thresh = nguong

        return split_idx, split_thresh

    def getInformationGain(self, y, cotHienTai, split_thresh):
        parentE = tinhE(y)
        conBenTraiIdxs, conBenPhaiIdxs = self._split(
            cotHienTai, split_thresh)

        if len(conBenTraiIdxs) == 0 or len(conBenPhaiIdxs) == 0:
            return 0

        n = len(y)
        numberLeft, numberRight = len(conBenTraiIdxs), len(conBenPhaiIdxs)
        eLeft, eRight = tinhE(y[conBenTraiIdxs]), tinhE(y[conBenPhaiIdxs])
        weightedAvg_ChildE = (numberLeft / n) * eLeft + \
            (numberRight / n) * eRight

        informationGain = parentE - weightedAvg_ChildE
        return informationGain

    def _split(self, X_column, split_thresh):
        # print(split_thresh)
        conBenTraiIdxs = np.argwhere(X_column <= split_thresh).flatten()
        conBenPhaiIdxs = np.argwhere(X_column >= split_thresh).flatten()
        return conBenTraiIdxs, conBenPhaiIdxs

    def _traverse_tree(self, x, node):
        if node.laNodeLa():
            return node.value

        if x[node.feature] <= node.nguong:
            return self._traverse_tree(x, node.conBenTrai)
        return self._traverse_tree(x, node.conBenPhai)

    # getMostCommonLabel
    def getLabel(self, y):
        counter = Counter(y)

        # if (len(counter) != 0):
        #     most_common = counter.most_common(1)[0][0]
        #     return most_common
        # else:
        #     return [0]

        label = counter.most_common(1)[0][0]
        return label


def bootstrap_sample(X, y):
    soLuongMau = X.shape[0]
    idxs = np.random.choice(soLuongMau, soLuongMau, replace=True)
    return X[idxs], y[idxs]


def most_commonumberLeftabel(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    def __init__(self, n_trees=3, min_samples_split=2, doSauToiDa=100, soLuongFeatures=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.doSauToiDa = doSauToiDa
        self.soLuongFeatures = soLuongFeatures
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                doSauToiDa=self.doSauToiDa,
                soLuongFeatures=self.soLuongFeatures,
            )
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_commonumberLeftabel(tree_pred)
                  for tree_pred in tree_preds]
        return np.array(y_pred)


if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = pd.read_csv("./stroke_final.csv")

    features = ["gender", "age", "hypertension"]

    X = data[features].to_numpy()[0: 1000]
    y = data["stroke"].to_numpy()[0: 1000]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_trees=3, doSauToiDa=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)
