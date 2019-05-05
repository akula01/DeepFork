from main import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import neighbors, linear_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import time
import parameters as params
from sklearn import neighbors, linear_model
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def get_roc():
    train_features, train_labels = get_train_data()
    test_features, test_labels = get_test_data()

    #Naive Bayes
    clf = MultinomialNB()
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    fpr_nbc, tpr_nbc, thresholds = roc_curve(test_labels, predictions)
    roc_auc_nbc = auc(fpr_nbc, tpr_nbc)

    #KNN
    clf = neighbors.KNeighborsClassifier()
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    fpr_knn, tpr_knn, thresholds = roc_curve(test_labels, predictions)
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    #Linear SVM
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    fpr_lsvc, tpr_lsvc, thresholds = roc_curve(test_labels, predictions)
    roc_auc_lsvc = auc(fpr_lsvc, tpr_lsvc)

    #RBF SVM
    clf = SVC(kernel='rbf')
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    fpr_rbf, tpr_rbf, thresholds = roc_curve(test_labels, predictions)
    roc_auc_rbf= auc(fpr_rbf, tpr_rbf)

    #Random Forest
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    fpr_rfc, tpr_rfc, thresholds = roc_curve(test_labels, predictions)
    roc_auc_rfc = auc(fpr_rfc, tpr_rfc)

    #Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    fpr_dtc, tpr_dtc, thresholds = roc_curve(test_labels, predictions)
    roc_auc_dtc = auc(fpr_dtc, tpr_dtc)

    #Extremely Randomized Tree
    clf = ExtraTreeClassifier()
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    fpr_ertc, tpr_ertc, thresholds = roc_curve(test_labels, predictions)
    roc_auc_ertc= auc(fpr_ertc, tpr_ertc)

    params.use_subset = True
    #DeepFork - Topological Features
    saved_dftopological = '/home/social-sim/PycharmProjects/Information_Diffusion/classifier_models/run_2_topological_adam_bs64_lr0.001_epochs_100/classifier.h5'
    dftopological = load_model(saved_dftopological)
    test_features, test_labels = get_test_data_topological()
    predictions = dftopological.predict(test_features)
    predictions[predictions > 0.5] = 1
    predictions[predictions < 0.5] = 0
    fpr_dftopological, tpr_dftopological, thresholds = roc_curve(test_labels, predictions)
    roc_auc_dftopological = auc(fpr_dftopological, tpr_dftopological)

    #DeepFork - Node Features
    params.with_fork = True
    saved_dfnode = '/home/social-sim/PycharmProjects/Information_Diffusion/classifier_models/run_adam_bs128_with_fork_lr0.001_6layers_epochs_100/classifier.h5'
    dfnode = load_model(saved_dfnode)
    test_features, test_labels = get_test_data()
    predictions = dfnode.predict(test_features)
    predictions[predictions > 0.5] = 1
    predictions[predictions < 0.5] = 0
    fpr_dfnode, tpr_dfnode, thresholds = roc_curve(test_labels, predictions)
    roc_auc_dfnode = auc(fpr_dfnode, tpr_dfnode)

    params.with_fork = False
    saved_dfnode_nowatch = '/home/social-sim/PycharmProjects/Information_Diffusion/classifier_models/run_adam_bs64_lr0.001_6layers_epochs_100/classifier.h5'
    dfnode_nowatch = load_model(saved_dfnode_nowatch)
    test_features, test_labels = get_test_data()
    predictions = dfnode_nowatch.predict(test_features)
    predictions[predictions > 0.5] = 1
    predictions[predictions < 0.5] = 0
    fpr_dfnode_nowatch, tpr_dfnode_nowatch, thresholds = roc_curve(test_labels, predictions)
    roc_auc_dfnode_nowatch= auc(fpr_dfnode_nowatch, tpr_dfnode_nowatch)

    params.with_fork = True
    saved_dfjoint = '/home/social-sim/PycharmProjects/Information_Diffusion/classifier_models/run_joint_adam_bs128_lr0.0001_6layers_epochs_100/classifier.h5'
    dfjoint = load_model(saved_dfjoint)
    test_features, test_labels = get_test_data_joint()
    predictions = dfjoint.predict(test_features)
    predictions[predictions > 0.5] = 1
    predictions[predictions < 0.5] = 0
    fpr_dfjoint, tpr_dfjoint, thresholds = roc_curve(test_labels, predictions)
    roc_auc_dfjoint= auc(fpr_dfjoint, tpr_dfjoint)

    plt.figure()
    lw = 2
    plt.plot(fpr_nbc, tpr_nbc, color='gray',
             lw=lw, label='Linear Naive Bayes ROC curve (area = %0.2f)' % roc_auc_nbc)
    plt.plot(fpr_knn, tpr_knn, color='darkorange',
         lw=lw, label='KNN ROC curve (area = %0.2f)' % roc_auc_knn)
    plt.plot(fpr_lsvc, tpr_lsvc, color='red',
             lw=lw, label='Linear SVM ROC curve (area = %0.2f)' % roc_auc_lsvc)
    plt.plot(fpr_rbf, tpr_rbf, color='green',
             lw=lw, label='RBF SVM ROC curve (area = %0.2f)' % roc_auc_rbf)
    plt.plot(fpr_rfc, tpr_rfc, color='blue',
             lw=lw, label='Random Forest ROC curve (area = %0.2f)' % roc_auc_rfc)
    plt.plot(fpr_dtc, tpr_dtc, color='yellow',
             lw=lw, label='Decision Tree ROC curve (area = %0.2f)' % roc_auc_dtc)
    plt.plot(fpr_ertc, tpr_ertc, color='brown',
             lw=lw, label='Extremely Randomized Tree ROC curve (area = %0.2f)' % roc_auc_ertc)
    plt.plot(fpr_dfnode, tpr_dfnode, color='cyan',
             lw=lw, label='DeepFork - Node - ROC curve (area = %0.2f)' % roc_auc_dfnode)
    plt.plot(fpr_dftopological, tpr_dftopological, color='pink',
             lw=lw, label='DeepFork - Node - No Watch - ROC curve (area = %0.2f)' % roc_auc_dftopological)
    plt.plot(fpr_dfnode_nowatch, tpr_dfnode_nowatch, color='magenta',
             lw=lw, label='DeepFork - Topological - ROC curve (area = %0.2f)' % roc_auc_dfnode_nowatch)
    plt.plot(fpr_dfjoint, tpr_dfjoint, color='lightblue',
             lw=lw, label='DeepFork - Node - Topological - Watch - ROC curve (area = %0.2f)' % roc_auc_dfjoint)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

get_roc()