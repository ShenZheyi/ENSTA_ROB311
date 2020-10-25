import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(test_label_pred, test_label_true, class_name, normalize=False):
    '''
    :param test_label_pred: 1D-ARRAY, predicted class
    :param test_label_true: 1D-ARRAY, actual class
    :param class_name: LIST, ['classname1','classname2',...]
    :param normalize: if plot confusion matrix in percentage
    :param cmap:
    '''
    conf_mat = confusion_matrix(test_label_true, test_label_pred)
    print('Confusion Matrix:\n', conf_mat)

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
    plt.savefig('confusion_matrix', dpi=200)


def read_data(filename, feature_cols, label_col):
    '''
    :param filename:
    :param feature_cols: LIST, serial numbers of columns representing features
    :param label_col: INT, serial number of columns representing label
    :param percent: FLOAT, percentage of data used to test
    :return: train_feature array, train_label array, test_feature array, test_label array
    '''
    print('Read', filename)
    feature = np.loadtxt(filename, delimiter=',', dtype=int, usecols=feature_cols)
    label = np.loadtxt(filename, delimiter=',', dtype=int, usecols=label_col)
    return feature, label


# Reference: https://stackoverflow.com/questions/45114760/how-to-plot-the-confusion-similarity-matrix-of-a-k-mean-algorithm
def label_transform(right_labels, pred_labels):
    k_labels = pred_labels  # Get cluster labels
    k_labels_matched = np.empty_like(k_labels)

    # For each cluster label...
    for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
        match_nums = [np.sum((k_labels == k) * (right_labels == t)) for t in np.unique(right_labels)]
        k_labels_matched[k_labels == k] = np.unique(right_labels)[np.argmax(match_nums)]

    return k_labels_matched


if __name__ == '__main__':
    training_file = "optdigits.tra"
    testing_file = "optdigits.tes"
    features_col = range(64)
    label_col = 64
    train_fea, train_lab = read_data(training_file, features_col, label_col)
    test_fea, test_lab_true = read_data(testing_file, features_col, label_col)

    # clustering without PCA
    kmeans_origin = KMeans(n_clusters=10, n_init=10, init='k-means++', random_state=0, max_iter=500).fit(train_fea)
    result_origin = kmeans_origin.predict(train_fea)
    result_origin = label_transform(train_lab, result_origin)
    plot_confusion_matrix(result_origin, train_lab, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], normalize=False)
    plot_confusion_matrix(result_origin, train_lab, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], normalize=True)
    print('The accuracy is:', np.mean(result_origin==train_lab))

    # clustering with PCA
    reduced_fea = PCA(n_components=2).fit_transform(train_fea)
    kmeans_pca = KMeans(n_clusters=10, n_init=10, init='k-means++', random_state=0, max_iter=500).fit(reduced_fea)
    result_pca = kmeans_pca.predict(reduced_fea)
    result_pca = label_transform(train_lab, result_pca)
    print('The accuracy is:', np.mean(result_pca == train_lab))

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_fea[:, 0].min() - 1, reduced_fea[:, 0].max() + 1
    y_min, y_max = reduced_fea[:, 1].min() - 1, reduced_fea[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans_pca.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_fea[:, 0], reduced_fea[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans_pca.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # use model to predict test data
    test_result = kmeans_origin.predict(test_fea)
    test_result = label_transform(test_lab_true, test_result)
    plot_confusion_matrix(test_result, test_lab_true, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], normalize=False)
    plot_confusion_matrix(test_result, test_lab_true, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], normalize=True)
    print('The accuracy is:', np.mean(test_result == test_lab_true))
