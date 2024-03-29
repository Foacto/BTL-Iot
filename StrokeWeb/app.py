from flask import Flask, request, render_template
from knn import Custom_KNN
import numpy as np
import mysql.connector
import pandas as pd
from bayes import NaiveBayes
import func
from decisiontree import *
from random_forest import CustomRandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Get db data
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="iot"
)

data = pd.read_sql("SELECT heart_disease, avg_glucose_level, age, Residence_type, \
    smoking_status, bmi, hypertension, work_type, gender, ever_married, \
    stroke FROM stroke_final LIMIT 1000", mydb)

X = None
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = None, None, None, None
choosed_feature = None
model = None

# Các var
cor_data = pd.read_sql("SELECT heart_disease, avg_glucose_level, age, \
    Residence_type, smoking_status, bmi, hypertension, work_type, gender, \
    ever_married FROM corr LIMIT 1000", mydb)

cor_heart_disease = round(cor_data['heart_disease'][0], 4)
cor_avg_glucose_level = round(cor_data['avg_glucose_level'][0], 4)
cor_age = round(cor_data['age'][0], 4)
cor_Residence_type = round(cor_data['Residence_type'][0], 4)
cor_smoking_status = round(cor_data['smoking_status'][0], 4)
cor_bmi = round(cor_data['bmi'][0], 4)
cor_hypertension = round(cor_data['hypertension'][0], 4)
cor_work_type = round(cor_data['work_type'][0], 4)
cor_gender = round(cor_data['gender'][0], 4)
cor_ever_married = round(cor_data['ever_married'][0], 4)

ip_listModel = ['KNN', 'Bayes', 'Decision Tree', 'Random Forest']
hd = ['Không', 'Có']
avg_glucose_level = 120
age = 20
rt = ['Thành thị', 'Nông thôn']
ss = ['Đã từng', 'Chưa bao giờ', 'Có', 'Không rõ']
bmi = 24
hp = ['Không', 'Có']
wt = ['Riêng tư', 'Tự kinh doanh', 'Cán bộ nhà nước', 'Trẻ con', 'Chưa đi làm']
gd = ['Nữ', 'Nam', 'Khác']
em = ['Đã kết hôn', 'Chưa kết hôn']

choosed_model = None
choosed_feature = None
table_name = None
table = None
accuracy = None

# KNN
max_heart_disease = data['heart_disease'].abs().max()
max_avg_glucose_level = data['avg_glucose_level'].abs().max()
max_age = data['age'].abs().max()
max_Residence_type = data['Residence_type'].abs().max()
max_smoking_status = data['smoking_status'].abs().max()
max_bmi = data['bmi'].abs().max()
max_hypertension = data['hypertension'].abs().max()
max_work_type = data['work_type'].abs().max()
max_gender = data['gender'].abs().max()
max_ever_married = data['ever_married'].abs().max()


@app.route('/')
def index():
    return render_template("home.html",
                           cor_hd=cor_heart_disease, cor_agl=cor_avg_glucose_level, cor_age=cor_age, cor_rt=cor_Residence_type, cor_st=cor_smoking_status, cor_bmi=cor_bmi, cor_h=cor_hypertension, cor_wt=cor_work_type, cor_gd=cor_gender, cor_em=cor_ever_married,
                           listModel=ip_listModel)


@app.route('/chooseFeatureM', methods=['POST', 'GET'])
def chooseFeatureM():
    global model, table_name, table, choosed_model, choosed_feature, accuracy, new_table

    choosed_feature = request.form.getlist('feature')

    X = data[choosed_feature]
    # print(X)

    if len(choosed_feature) == 0:
        return render_template("home.html",
                               cor_hd=cor_heart_disease, cor_agl=cor_avg_glucose_level, cor_age=cor_age, cor_rt=cor_Residence_type, cor_st=cor_smoking_status, cor_bmi=cor_bmi, cor_h=cor_hypertension, cor_wt=cor_work_type, cor_gd=cor_gender, cor_em=cor_ever_married,
                               listModel=ip_listModel, message='Choose atleast 1 feature before submit!')

    choosed_model = request.form.get('modelselect')

    sk_model = None
    if choosed_model == 'KNN':
        sk_model = KNeighborsClassifier(n_neighbors=7)
        model = Custom_KNN(k=7)
        X_train, X_test, y_train, y_test = func.train_test_split_scratch(
            X, y, test_size=0.2, shuffle=True)
        X_train = func.normalize(X_train, columns=choosed_feature)
        model.fit(X=X_train, y=y_train)
    if choosed_model == 'Bayes':
        sk_model = GaussianNB()
        model = NaiveBayes()
        X_train, X_test, y_train, y_test = func.train_test_split_scratch(
            X, y, test_size=0.2, shuffle=True)
        model.fit(X=X_train, y=y_train)
    if choosed_model == 'Decision Tree':
        sk_model = DecisionTreeClassifier(max_depth=10)
        model = DecisionTree(chieu_sau_toida=10)
        X_train, X_test, y_train, y_test = func.train_test_split_scratch(
            X, y, test_size=0.2, shuffle=True)
        model.fit(X=X_train, y=y_train)

    if choosed_model == 'Random Forest':
        sk_model = RandomForestClassifier(
            n_estimators=20, max_depth=10, min_samples_leaf=2)

        model = CustomRandomForest(n_decisiontrees=20, doSauToiDa=10)
        X_train, X_test, y_train, y_test = func.train_test_split_scratch(
            X, y, test_size=0.2, shuffle=True)
        model.fit(X_train, y_train)

    tmpX = X_test.copy()
    if choosed_model == 'KNN':
        tmpX = func.normalize(tmpX, columns=choosed_feature)

    sk_model.fit(X_train, y_train)
    ske_pred = sk_model.predict(tmpX)
    pred = model.predict(tmpX)

    print('\nAccuracy of sklearn:', round(
        func.accuracy(y_test, ske_pred) * 100), 'percent.\n')

    accuracy = round(func.accuracy(y_test, pred) * 100)
    f1_score = round(func.f1(y_test, pred) * 100)
    recall = round(func.recall(y_test, pred) * 100)
    precision = round(func.precision(y_test, pred) * 100)
    func.add_f1(f1_score)
    func.add_acc(accuracy)
    func.add_recall(recall)
    func.add_precision(precision)

    table_name = choosed_feature
    table_name.append('actual')
    table_name.append('predict')
    table = X_test
    new_table = []
    for i in range(len(pred)):
        temp = np.append(table[i], y_test[i])
        new_table.append(np.append(temp, pred[i]))

    return render_template("home.html",
                           cor_hd=cor_heart_disease, cor_agl=cor_avg_glucose_level, cor_age=cor_age, cor_rt=cor_Residence_type, cor_st=cor_smoking_status, cor_bmi=cor_bmi, cor_h=cor_hypertension, cor_wt=cor_work_type, cor_gd=cor_gender, cor_em=cor_ever_married,
                           listModel=ip_listModel, accuracy=accuracy, model=choosed_model,
                           choosed_feature=choosed_feature,
                           hd=hd, aglip=avg_glucose_level, ageip=age, rt=rt, ss=ss, bmiip=bmi, hp=hp, wt=wt, gd=gd, em=em,
                           table_name=table_name, data_table=new_table)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    global data, hd, avg_glucose_level, age, rt, ss, bmi, hp, wt, gd, em

    var_X = []

    for i in choosed_feature:
        if i == 'heart_disease':
            selecthd = request.form.get('hdselect')
            if selecthd == 'Có':
                tmp = 1
                if choosed_model == 'KNN':
                    tmp /= max_heart_disease
                var_X.append(tmp)
                hd = ['Có', 'Không']
            else:
                tmp = 0
                if choosed_model == 'KNN':
                    tmp /= max_heart_disease
                var_X.append(tmp)
                hd = ['Không', 'Có']
        elif i == 'avg_glucose_level':
            avg_glucose_level = float(request.form['aglip'])
            tmp = avg_glucose_level
            if choosed_model == 'KNN':
                tmp /= max_avg_glucose_level

            var_X.append(tmp)
        elif i == 'age':
            age = int(request.form['ageip'])
            tmp = age
            if choosed_model == 'KNN':
                tmp /= max_age

            var_X.append(tmp)
        elif i == 'Residence_type':
            selectrt = request.form.get('rtselect')
            if selectrt == 'Nông thôn':
                tmp = 1
                if choosed_model == 'KNN':
                    tmp /= max_Residence_type
                var_X.append(tmp)
                rt = ['Nông thôn', 'Thành thị']
            else:
                tmp = 0
                if choosed_model == 'KNN':
                    tmp /= max_Residence_type
                var_X.append(tmp)
                rt = ['Thành thị', 'Nông thôn']
        elif i == 'smoking_status':
            selectss = request.form.get('ssselect')
            if selectss == 'Đã từng':
                tmp = 0
                if choosed_model == 'KNN':
                    tmp /= max_smoking_status
                var_X.append(tmp)
                ss = ['Đã từng', 'Chưa bao giờ', 'Có', 'Không rõ']
            elif selectss == 'Chưa bao giờ':
                tmp = 1
                if choosed_model == 'KNN':
                    tmp /= max_smoking_status
                var_X.append(tmp)
                ss = ['Chưa bao giờ', 'Đã từng', 'Có', 'Không rõ']
            elif selectss == 'Có':
                tmp = 2
                if choosed_model == 'KNN':
                    tmp /= max_smoking_status
                var_X.append(tmp)
                ss = ['Có', 'Đã từng', 'Chưa bao giờ', 'Không rõ']
            elif selectss == 'Không rõ':
                tmp = 3
                if choosed_model == 'KNN':
                    tmp /= max_smoking_status
                var_X.append(tmp)
                ss = ['Không rõ', 'Chưa bao giờ', 'Đã từng', 'Có']
        elif i == 'bmi':
            bmi = float(request.form['bmiip'])
            tmp = bmi
            if choosed_model == 'KNN':
                tmp /= max_bmi

            var_X.append(bmi)
        elif i == 'hypertension':
            selecthp = request.form.get('hselect')
            if selecthp == 'Có':
                tmp = 1
                if choosed_model == 'KNN':
                    tmp /= max_hypertension
                var_X.append(tmp)
                hp = ['Có', 'Không']
            else:
                tmp = 0
                if choosed_model == 'KNN':
                    tmp /= max_hypertension
                var_X.append(tmp)
                hp = ['Không', 'Có']
        elif i == 'work_type':
            selectwt = request.form.get('wtselect')
            if selectwt == 'Riêng tư':
                tmp = 0
                if choosed_model == 'KNN':
                    tmp /= max_work_type
                var_X.append(tmp)
                wt = ['Riêng tư', 'Tự kinh doanh',
                      'Cán bộ nhà nước', 'Trẻ con', 'Chưa đi làm']
            elif selectwt == 'Tự kinh doanh':
                tmp = 1
                if choosed_model == 'KNN':
                    tmp /= max_work_type
                var_X.append(tmp)
                wt = ['Tự kinh doanh', 'Riêng tư',
                      'Cán bộ nhà nước', 'Trẻ con', 'Chưa đi làm']
            elif selectwt == 'Cán bộ nhà nước':
                tmp = 2
                if choosed_model == 'KNN':
                    tmp /= max_work_type
                var_X.append(tmp)
                wt = ['Cán bộ nhà nước', 'Riêng tư',
                      'Tự kinh doanh', 'Trẻ con', 'Chưa đi làm']
            elif selectwt == 'Trẻ con':
                tmp = 3
                if choosed_model == 'KNN':
                    tmp /= max_work_type
                var_X.append(tmp)
                wt = ['Trẻ con', 'Riêng tư', 'Tự kinh doanh',
                      'Cán bộ nhà nước', 'Chưa đi làm']
            elif selectwt == 'Chưa đi làm':
                tmp = 4
                if choosed_model == 'KNN':
                    tmp /= max_work_type
                var_X.append(tmp)
                wt = ['Chưa đi làm', 'Riêng tư', 'Tự kinh doanh',
                      'Cán bộ nhà nước', 'Trẻ con']
        elif i == 'gender':
            selectgd = request.form.get('gdselect')
            if selectgd == 'Nữ':
                tmp = 0
                if choosed_model == 'KNN':
                    tmp /= max_gender
                var_X.append(tmp)
                gd = ['Nữ', 'Nam', 'Khác']
            elif selectgd == 'Nam':
                tmp = 1
                if choosed_model == 'KNN':
                    tmp /= max_gender
                var_X.append(tmp)
                gd = ['Nam', 'Nữ', 'Khác']
            else:
                tmp = 2
                if choosed_model == 'KNN':
                    tmp /= max_gender
                var_X.append(tmp)
                gd = ['Khác', 'Nữ', 'Nam']
        elif i == 'ever_married':
            selectem = request.form.get('emselect')
            if selectem == 'Đã kết hôn':
                tmp = 1
                if choosed_model == 'KNN':
                    tmp /= max_ever_married
                var_X.append(tmp)
                em = ['Đã kết hôn', 'Chưa kết hôn']
            else:
                tmp = 0
                if choosed_model == 'KNN':
                    tmp /= max_ever_married
                var_X.append(tmp)
                em = ['Chưa kết hôn', 'Đã kết hôn']

    predict = model.predict([var_X])
    predict = int(predict[0])

    if predict == 1:
        kq = "Có khả năng bị đột quỵ!"
    else:
        kq = "Không có khả năng bị đột quỵ!"

    print("\n", kq, "\n")

    return render_template("home.html",
                           cor_hd=cor_heart_disease, cor_agl=cor_avg_glucose_level, cor_age=cor_age, cor_rt=cor_Residence_type, cor_st=cor_smoking_status, cor_bmi=cor_bmi, cor_h=cor_hypertension, cor_wt=cor_work_type, cor_gd=cor_gender, cor_em=cor_ever_married,
                           listModel=ip_listModel, accuracy=accuracy, model=choosed_model,
                           choosed_feature=choosed_feature,
                           hd=hd, aglip=avg_glucose_level, ageip=age, rt=rt, ss=ss, bmiip=bmi, hp=hp, wt=wt, gd=gd, em=em, table=table,
                           table_name=table_name, ketqua=kq, data_table=new_table)


if __name__ == "__main__":
    app.run()
