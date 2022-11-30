import numpy as np
import pandas as pd
import math

def shuffle_data(X, y):
    Data_num = np.arange(X.shape[0])
    np.random.shuffle(Data_num)

    return X[Data_num], y[Data_num]

def train_test_split_scratch(X, y, test_size=0.5, shuffle=True):
    if shuffle:
        X, y = shuffle_data(X.values, y.values)
        if test_size <1 :
            train_ratio = len(y) - int(len(y) *test_size)
            X_train, X_test = X[:train_ratio], X[train_ratio:]
            y_train, y_test = y[:train_ratio], y[train_ratio:]
            return X_train, X_test, y_train, y_test
    else:
        if test_size <1 :
            train_ratio = len(y) - int(len(y) *test_size)
            X_train, X_test = X[:train_ratio].values, X[train_ratio:].values
            y_train, y_test = y[:train_ratio].values, y[train_ratio:].values
            return X_train, X_test, y_train, y_test

def normalize(X, columns):
    tmp = pd.DataFrame(X, columns=columns)
    for column in tmp.columns:
        tmp[column] = tmp[column]  / tmp[column].abs().max()

    return tmp.values

def cov(a, b):
    result = 0
    for i in a.keys():
        result += (a[i] - np.mean(a))*(b[i] - np.mean(b))

    result /= (len(b) - 1)

    return result

def varr(a):
    result = 0
    for i in a.keys():
        result += (a[i] - np.mean(a)) ** 2

    result /= (len(a) - 1)

    return abs(result)

def Ccorr(a,b):
    result = cov(a,b) / (math.sqrt(varr(a)) * math.sqrt(varr(b)))

    return result

def accuracy(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)  
    fp = false_positive(ground_truth, prediction)  
    fn = false_negative(ground_truth, prediction)  
    tn = true_negative(ground_truth, prediction)  
    acc_score = (tp + tn)/ (tp + tn + fp + fn)  
    return acc_score

def precision(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)  
    fp = false_positive(ground_truth, prediction)  
    prec = tp/ (tp + fp)  
    return prec

def recall(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)  
    fn = false_negative(ground_truth, prediction)  
    prec = tp/ (tp + fn)  
    return prec

def f1(ground_truth, prediction):
    p = precision(ground_truth, prediction)
    r = recall(ground_truth, prediction)
    f1_score = 2 * p * r/ (p + r) 
    return f1_score

#True positive
def true_positive(ground_truth, prediction):
    tp = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 1 and pred == 1:
            tp +=1
    return tp

#False Positive
def false_positive(ground_truth, prediction):
    fp = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 0 and pred == 1:
            fp +=1
    return fp

#True Negative
def true_negative(ground_truth, prediction):
    tn = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 0 and pred == 0:
            tn +=1
    return tn

#False Negative
def false_negative(ground_truth, prediction):
    fn = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 1 and pred == 0:
            fn +=1
    return fn