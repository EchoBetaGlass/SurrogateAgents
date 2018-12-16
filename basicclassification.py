"""Conduct basic classification independent of voting."""
import pandas as pd
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neighbors.nearest_centroid import NearestCentroid as NC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import ExtraTreeClassifier as ExTC1
from sklearn.ensemble import ExtraTreesClassifier as ExTC2
from sklearn.neural_network import MLPClassifier as NNC
from sklearn.metrics import confusion_matrix

classifiers = {
    "BC": BC,
    "SVC": SVC,
    "KNC": KNC,
    "NC": NC,
    "GPC": GPC,
    "DTC": DTC,
    "NNC": NNC,
    "ExTC1": ExTC1,
    "ExTC2": ExTC2,
}
training_costs = pd.DataFrame(columns=classifiers.keys())
prediction_costs = pd.DataFrame(columns=classifiers.keys())

data = pd.read_csv("features with R2 dtlz.csv")
inputs_train = data[data.keys()[1:11]]
outputs_train = data[data.keys()[11:]]
target_train = outputs_train.idxmax(axis=1)

data_wfg = pd.read_csv("features with R2 wfg.csv")
inputs_test = data_wfg[data_wfg.keys()[1:11]]
outputs_test = data_wfg[data_wfg.keys()[11:]]
target_test = outputs_test.idxmax(axis=1)
for i in range(50):
    temp_t = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=classifiers.keys())
    temp_p = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=classifiers.keys())
    for key, value in classifiers.items():
        # DTLZ STUFFS
        clf = value()
        clf = clf.fit(inputs_train, target_train)
        predictions = clf.predict(inputs_train)
        con_mat = confusion_matrix(target_train, predictions)
        cost_mat = outputs_train
        cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
        # print(con_mat)
        num_samples, num_classes = cost_mat.shape
        cost = 0
        lenp = len(predictions)
        for index in range(num_samples):
            cost += cost_mat.iloc[index][predictions[index]]
        # print("Cost during training", cost / lenp)
        cost = cost / lenp
        temp_t[key] = cost
        # WFG STUFFS

        predictions = clf.predict(inputs_test)
        con_mat = confusion_matrix(target_test, predictions)
        cost_mat = outputs_test
        cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
        # print(con_mat)
        num_samples, num_classes = cost_mat.shape
        cost = 0
        lenp = len(predictions)
        for index in range(num_samples):
            cost += cost_mat.iloc[index][predictions[index]]
        # print("Cost during testing", cost / lenp)
        temp_p[key] = cost / lenp
    training_costs = training_costs.append(temp_t)
    prediction_costs = prediction_costs.append(temp_p)
print(training_costs)
print(prediction_costs)
training_costs.to_csv("training costs.csv")
prediction_costs.to_csv("prediction costs.csv")
