from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np


def tinhE(y):
    # y = [0, 1, 0] --> soLuongCacNhan = [2, 1]
    soLuongCacNhan = np.bincount(y)
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

    def duDoan(self, X):
        return np.array([self.duyetCay(x, self.nodeGoc) for x in X])

    def phatTrienCay(self, X, y, doSauHienTai=0):
        soLuongMau, soLuongFeatures = X.shape
        soLuongNhan = len(np.unique(y))

        if (
            doSauHienTai >= self.doSauToiDa
            or soLuongNhan == 1
            or soLuongMau < self.min_samples_split
        ):
            leafValue = self.getValueHere(y)
            return Node(value=leafValue)

        featureIndexs = np.random.choice(
            soLuongFeatures, self.soLuongFeatures, replace=False)

        best_split_feat_index, best_nguong = self.getBestFeatIdxAndSplitThresh(
            X, y, featureIndexs)

        nhoHonNguongIdxs, lonHonNguongIdxs = self.phanTach(
            X[:, best_split_feat_index], best_nguong)

        # X[nhoHonNguongIdxs, :] --> các mẫu có giá trị tại split_feature <= best_nguong
        conBenTrai = self.phatTrienCay(
            X[nhoHonNguongIdxs, :], y[nhoHonNguongIdxs], doSauHienTai + 1)
        conBenPhai = self.phatTrienCay(
            X[lonHonNguongIdxs, :], y[lonHonNguongIdxs], doSauHienTai + 1)

        return Node(best_split_feat_index, best_nguong, conBenTrai, conBenPhai)

    def getBestFeatIdxAndSplitThresh(self, X, y, featureIndexs):
        doLoiThongTinCaoNhat = -1
        phanTachIdx, nguongTachTotNhat = None, None

        for feat_idx in featureIndexs:
            cotHienTai = X[:, feat_idx]
            nguongs = np.unique(cotHienTai)

            for nguong in nguongs:
                doLoiThongTin = self.getInformationGain(y, cotHienTai, nguong)

                if doLoiThongTin > doLoiThongTinCaoNhat:
                    doLoiThongTinCaoNhat = doLoiThongTin
                    phanTachIdx = feat_idx
                    nguongTachTotNhat = nguong

        return phanTachIdx, nguongTachTotNhat

    def getInformationGain(self, y, cotHienTai, nguongTach):
        parentE = tinhE(y)
        nhoHonNguongIdxs, lonHonNguongIdxs = self.phanTach(
            cotHienTai, nguongTach)

        n = len(y)
        soLuongNhoHonNguong, soLuongLonHonNguong = len(
            nhoHonNguongIdxs), len(lonHonNguongIdxs)
        eLeft, eRight = tinhE(y[nhoHonNguongIdxs]), tinhE(
            y[lonHonNguongIdxs])
        weightedAvg_childE = (soLuongNhoHonNguong / n) * eLeft + \
            (soLuongLonHonNguong / n) * eRight

        informationGain = parentE - weightedAvg_childE
        return informationGain

    def phanTach(self, X_column, nguongTach):
        nhoHonNguongIdxs = np.argwhere(X_column <= nguongTach).flatten()
        lonHonNguongIdxs = np.argwhere(X_column >= nguongTach).flatten()
        return nhoHonNguongIdxs, lonHonNguongIdxs

    def duyetCay(self, x, node):
        if node.laNodeLa():
            return node.value

        if x[node.feature] <= node.nguong:
            return self.duyetCay(x, node.conBenTrai)
        return self.duyetCay(x, node.conBenPhai)

    # getMostCommonLabel
    def getValueHere(self, y):
        counter = Counter(y)
        label = counter.most_common(1)[0][0]
        return label


def randomSamples(X, y):
    soLuongMau = X.shape[0]
    randomIdxs = np.random.choice(soLuongMau, soLuongMau, replace=True)
    return X[randomIdxs], y[randomIdxs]


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
            X_samp, y_samp = randomSamples(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def duDoan(self, X):
        tree_preds = np.array([tree.duDoan(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [self.getValueHere(tree_pred)
                  for tree_pred in tree_preds]
        return np.array(y_pred)

    def getValueHere(self, y):
        counter = Counter(y)
        label = counter.most_common(1)[0][0]
        return label


if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = pd.read_csv("./stroke_final.csv")

    features = ["gender", "age", "hypertension", "smoking_status"]

    X = data[features].to_numpy()[0: 1000]
    y = data["stroke"].to_numpy()[0: 1000]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_trees=3, doSauToiDa=10)

    clf.fit(X_train, y_train)
    y_pred = clf.duDoan(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)
