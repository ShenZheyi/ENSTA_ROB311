import numpy as np
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split

def KNN_classifier(train_feature, train_label, test_feature, k):
    '''
    :param train_festure: 2D-ARRAY, train data number * feature number
    :param train_label: 1D-ARRAY, 1 * train data number
    :param test_feature: 2D-ARRAY, test data number * feature number
    :param k: INT, number of nearest neighbors
    :return: test_label, 1D-ARRAY, 1 * test data number, prediction class for each sample
    '''
    train_count = train_feature.shape[0]  # number of train data
    test_count = test_feature.shape[0]  # number of test data

    distances = np.zeros((test_count, train_count))
    for i in range(test_count):
        for j in range(train_count):
            # the distance between the i-st item in the test and the j-st item in the dataset
            distances[i][j] = math.sqrt(np.sum((test_feature[i] - train_feature[j]) ** 2))

    sort_index = np.zeros((test_count, train_count))
    for i in range(test_count):
        sort_index[i] = np.argsort(distances[i])  # return the index of sorted results
    sort_index = sort_index.astype(int)

    k_neighbors = np.zeros((test_count, k))
    for i in range(test_count):
        for j in range(k):
            k_neighbors[i][j] = train_label[sort_index[i][j]]

    test_label_pred = np.zeros(test_count)
    for i in range(test_count):
        test_label_pred[i] = Counter(k_neighbors[i]).most_common(1)[0][0]
    test_label_pred = test_label_pred.astype(int)

    return test_label_pred

def calculate_accuracy(test_label_pred, test_label_true):
    return np.sum(test_label_pred == test_label_true)/test_label_true.shape[0]

def plot_confusion_matrix(test_label_pred, test_label_true, class_name, normalize=False):
    '''
    :param test_label_pred: 1D-ARRAY, predicted class
    :param test_label_true: 1D-ARRAY, actual class
    :param class_name: LIST, ['classname1','classname2',...]
    :param normalize: if plot confusion matrix in percentage
    :param cmap:
    '''
    conf_mat = confusion_matrix(test_label_true, test_label_pred)
    print('Confusion Matrix:\n',conf_mat)

    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix:\n', conf_mat)

    title = 'Confusion matrix in percentage' if normalize else 'Confusion matrix'
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_name))
    plt.xticks(tick_marks, class_name)
    plt.yticks(tick_marks, class_name, rotation=90)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.show()
    plt.savefig('confusion_matrix',dpi=200)

def read_data(filename, feature_cols, label_col, percent=0.2):
    '''
    :param filename:
    :param feature_cols: LIST, serial numbers of columns representing features
    :param label_col: INT, serial number of columns representing label
    :param percent: FLOAT, percentage of data used to test
    :return: train_feature array, train_label array, test_feature array, test_label array
    '''
    data = np.loadtxt(filename, delimiter=',', dtype=np.str)
    data = np.delete(data, np.where(data=='?')[0], axis=0).astype(int)
    train_data, test_data, train_index, test_index = train_test_split(data, range(data.shape[0]), test_size=percent)
    train_feature = train_data[:, feature_cols]
    train_label = train_data[:, label_col]
    test_feature = test_data[:, feature_cols]
    test_label = test_data[:, label_col]

    return train_feature, train_label, test_feature, test_label

if __name__ == '__main__':
    k_values = [1, 3, 5, 7, 9, 11, 15, 19, 25, 31, 39]
    num = 50 # test number for each k value

    # breast cancer wisconsin
    breast_cancer_file = 'breast-cancer-wisconsin.data'
    fea_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    lab_col = 10
    accuracy_list_1 = []

    for k in k_values:
        acc = np.zeros(num)
        for t in range(num):
            train_fea, train_lab, test_fea, test_lab_true = read_data(breast_cancer_file, fea_cols, lab_col)
            test_lab_pred = KNN_classifier(train_fea, train_lab, test_fea, k)
            # calculate accuracy
            acc[t] = calculate_accuracy(test_lab_pred, test_lab_true)
            print('The accuracy of kNN algorithm for case of breast cancer wisconsin is', acc[t])
        accuracy_list_1.append(np.mean(acc))

    print(accuracy_list_1)
    plt.plot(k_values, accuracy_list_1, 'bo-')
    plt.title('Accuracy of kNN algorithm with different k value')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

    # Haberman
    haberman_file = 'haberman.data'
    fea_cols = [0,1, 2]
    lab_col = 3
    accuracy_list_2 = []

    for k in k_values:
        acc = np.zeros(num)
        for t in range(num):
            train_fea, train_lab, test_fea, test_lab_true = read_data(haberman_file, fea_cols, lab_col)
            test_lab_pred = KNN_classifier(train_fea, train_lab, test_fea, k)
            # calculate accuracy
            acc[t] = calculate_accuracy(test_lab_pred, test_lab_true)
            print('The accuracy of kNN algorithm for case of breast cancer wisconsin is', acc[t])
        accuracy_list_2.append(np.mean(acc))

    print(accuracy_list_2)
    plt.plot(k_values, accuracy_list_2, 'bo-')
    plt.title('Accuracy of kNN algorithm with different k value')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

