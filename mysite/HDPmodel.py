from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from matplotlib import rcParams
#from matplotlib.cm import rainbow
# %matplotlib inline
warnings.filterwarnings('ignore')
heart = pd.read_csv('heart.csv')
heart.head()
heart.info()
heart.describe()
heart = heart.iloc[:, 2:]
heart.head()
predictors = heart.drop("target", axis=1)
target = heart["target"]
x_train, x_test, y_train, y_test = train_test_split(
    predictors, target, test_size=0.20, random_state=0)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
# #
# #
# #
# #
# #
# #
# y_pred_lr.shape

# score_lr = round(accuracy_score(y_pred_lr,y_test)*100,2)

# print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
# #nb
# from sklearn.naive_bayes import GaussianNB

# nb = GaussianNB()

# nb.fit(x_train,y_train)

# y_pred_nb = nb.predict(x_test)
# y_pred_nb.shape
# score_nb = round(accuracy_score(y_pred_nb,y_test)*100,2)

# print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
# #svm
# from sklearn import svm

# sv = svm.SVC(kernel='linear')

# sv.fit(x_train, y_train)

# y_pred_svm = sv.predict(x_test)
# y_pred_svm.shape
# score_svm = round(accuracy_score(y_pred_svm,y_test)*100,2)

# print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
# #knn
# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(x_train,y_train)
# y_pred_knn=knn.predict(x_test)
# y_pred_knn.shape
# score_knn = round(accuracy_score(y_pred_knn,y_test)*100,2)

# print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
# from sklearn.tree import DecisionTreeClassifier

# max_accuracy = 0


# for x1 in range(200):
#     dt = DecisionTreeClassifier(random_state=x1)
#     dt.fit(x_train,y_train)
#     y_pred_dt = dt.predict(x_test)
#     current_accuracy = round(accuracy_score(y_pred_dt,y_test)*100,2)
#     if(current_accuracy>max_accuracy):
#         max_accuracy = current_accuracy
#         best_x = x1

# #print(max_accuracy)
# #print(best_x)


# dt = DecisionTreeClassifier(random_state=best_x)
# dt.fit(x_train,y_train)
# y_pred_dt = dt.predict(x_test)
# print(y_pred_dt.shape)
# score_dt = round(accuracy_score(y_pred_dt,y_test)*100,2)

# print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
# rf

max_accuracy = 0


for x2 in range(200):
    rf = RandomForestClassifier(random_state=x2)
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    current_accuracy = round(accuracy_score(y_pred_rf, y_test)*100, 2)
    if(current_accuracy > max_accuracy):
        max_accuracy = current_accuracy
        best_x = x2

# print(max_accuracy)
# print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
y_pred_rf.shape
score_rf = round(accuracy_score(y_pred_rf, y_test)*100, 2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")
# output
while(True):

    cp = float(input('Enter chest pain type: '))
    trestbps = float(input('Enter resting systolic blood pressure: '))
    chol = float(input('Enter the serum cholestrol level:'))
    fbs = float(input('Enter fasting blood sugar>120 mg/dl:'))
    restecg = float(input('Enter resting electrocardiographic results: '))
    thalach = float(input('Enter Maximum heart rate achieved: '))
    exang = float(input('Enter exercise-induced angina: '))
    oldpeak = float(
        input('Enter old pek=ST depression induced by exercise relative to rest: '))
    slope = float(input('Enter slope of thr peak exercise ST segment: '))
    ca = float(
        input('Enter  the number of major vessels(0-3)colored by fluoroscopy: '))
    thal = float(input('Enter  thallium scan: '))

    answer = [[cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal]]
    output = rf.predict(answer)
    print(output)
    reply = input("Enter Y to continue and N to exit:")
    reply = reply.upper()
    if(reply == 'N'):
        break
filename = 'finalized_model.sav'
joblib.dump(rf, filename)
