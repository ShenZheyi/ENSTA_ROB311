# -*- coding: utf-8 -*-
"""ROB311-TP4-SVM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KuMfVATrfjBrmf0SzcMlgNynVC207S9f

# **ROB311-TP4-SVM**
## Implementation of a SVM Digit Recognition Algorithm
SHEN Zheyi & GUAN Zhaoyi
"""

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler 
import numpy as np

# Fontion used to read data:

def read_data(filename, feature_cols, label_cols):
  print('Read', filename)
  feature = np.loadtxt(filename, delimiter=',', dtype=int, usecols=feature_cols, skiprows=1)
  label = np.loadtxt(filename, delimiter=',', dtype=int, usecols=label_cols, skiprows=1)
  return feature, label

# Fontions to calculate the accuracy and the confusion matrix:

def calculate_accuracy(test_label_true, test_label_pred):
  accuracy = np.mean(test_label_true == test_label_pred)
  print('The accuray of SVM model is: ', accuracy)
  return accuracy

def calculate_confusion_matrix(test_label_true, test_label_pred):
  conf_mat = confusion_matrix(test_label_true, test_label_pred)
  print('The confusion Matrix :')
  print(conf_mat)
  return conf_mat

""" 1. A Simple Implementation of SVM"""

def simple_SVM_train(train_label, train_data, pca_components=100):
  # Create and train the model
  pca = PCA(n_components=pca_components)
  pca.fit(train_data)
  new_train_data = pca.transform(train_data)
  print('Size of train data before PCA: ', train_data.shape)
  print('Size of train data after PCA: ', new_train_data.shape)
  clf = SVC()
  print('Begin training...')
  clf.fit(new_train_data, train_label)
  return pca, clf

def simple_SVM_test(test_data, pca, clf):
  new_test_data = pca.transform(test_data)
  print('Begin testing...')
  predict_label = clf.predict(new_test_data)
  return predict_label

"""2. Algorithm using *make_pipeline*"""

def SVM_train_pipeline(train_label, train_data, pca_components=100):
  # Create and train the model
  pca = PCA(n_components=pca_components)
  svc = SVC(class_weight='balanced')
  clf = make_pipeline(pca, StandardScaler(), svc)
  print('Begin training...')
  clf.fit(train_data, train_label)
  return clf

def SVM_test_pipeline(test_data, clf):
  print('Begin testing...')
  predict_label = clf.predict(test_data)
  return predict_label

"""3. Use GridSearchCV to find best estimator"""

def SVM_train(train_label, train_data, pca_components=100):
  # Create and train the model
  pca = PCA(n_components=pca_components)
  svc = SVC(class_weight='balanced')
  model = make_pipeline(pca, StandardScaler(), svc)
  parameters = {'svc__C': [1, 5, 10],'svc__kernel': ('linear', 'poly', 'rbf', 'sigmoid')}
  clf = GridSearchCV(model, parameters)
  print('Begin training...')
  clf.fit(train_data, train_label)
  print("The best parameters is : ", clf.best_params_)
  print("The best score associated is : ", clf.best_score_)
  return clf.best_estimator_

def SVM_test(test_data, estimator):
  print('Begin test...')
  predict_label = estimator.predict(test_data)
  return predict_label
  
if __name__ == '__main__':
  train_file = "mnist_train.csv"
  test_file = "mnist_test.csv"
  label_col = 0
  features_col = range(1, 785)
  trainData, trainLabel = read_data(train_file, features_col, label_col)
  testData, trueLabel = read_data(test_file, features_col, label_col)

  print('First test, simple implementation')
  pca, clf_simple = simple_SVM_train(trainLabel, trainData, 100)
  predictLabel = simple_SVM_test(testData, pca, clf_simple)
  calculate_accuracy(trueLabel, predictLabel)
  calculate_confusion_matrix(trueLabel, predictLabel)

  print('Implementation with make_pipeline')
  clf_pipeline = SVM_train_pipeline(trainLabel, trainData, 100)
  predictLabel = SVM_test_pipeline(testData, clf_pipeline)
  calculate_accuracy(trueLabel, predictLabel)
  calculate_confusion_matrix(trueLabel, predictLabel)

  print('Implementation with GirdSearchCV to find best estimator')
  best_estimator = SVM_train(trainLabel, trainData)
  predictLabel = SVM_test(testData, best_estimator)
  calculate_accuracy(trueLabel, predictLabel)
  calculate_confusion_matrix(trueLabel, predictLabel)