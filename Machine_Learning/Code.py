1.#Logistic Regression
import numpy as np
import matplotlib as plt
from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model, datasets

dataset = datasets.load_iris()
logreg = linear_model.LogisticRegression(C=1e5)
model = logreg.fit(dataset.data, dataset.target)
expected = dataset.target
predicted = model.predict(dataset.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
