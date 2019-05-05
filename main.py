from utils.data_util import *
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
from keras.models import load_model
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
import pickle
import random

def build_model(input_size):
    model = Sequential([
        BatchNormalization(input_shape=(input_size,)),
        #Dense(32, activation='relu'),
        #Dense(16, activation='relu'),
        Dense(input_size, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def get_train_data_topological():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_positive_features_topological.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_negative_features_topological.pkl', 'rb'))
    samples = np.concatenate((positive_samples, negative_samples))
    features, labels = [], []
    for sample in samples:
        if sample[0] is None or sample[1] is None:
            continue
        if params.use_subset:
            features.append(sample[0][0:12])
        else:
            features.append(sample[0])
        labels.append(sample[1])

    indexes = list(range(len(features)))
    random.shuffle(indexes)
    features = [features[i] for i in indexes][0:params.train_count]
    labels = [labels[i] for i in indexes][0:params.train_count]

    return np.array(features), np.array(labels)

def get_train_data_joint():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_positive_features_joint.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_negative_features_joint.pkl', 'rb'))
    samples = np.concatenate((positive_samples, negative_samples))
    features, labels = [], []
    for sample in samples:
        if sample[0] is None or sample[1] is None:
            continue
        features.append(sample[0][0:29])
        labels.append(sample[1])

    indexes = list(range(len(features)))
    random.shuffle(indexes)
    features = [features[i] for i in indexes][0:params.train_count]
    labels = [labels[i] for i in indexes][0:params.train_count]

    return np.array(features), np.array(labels)

def get_train_data():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_positive_features.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/train_data_negative_features.pkl', 'rb'))
    samples = np.concatenate((positive_samples, negative_samples))
    features, labels = [], []
    for sample in samples:
        if sample[0] is None or sample[1] is None:
            continue
        if params.with_fork:
            features.append(sample[0])
        else:
            features.append(sample[0][:15])
        labels.append(sample[1])

    indexes = list(range(len(features)))
    random.shuffle(indexes)
    features = [features[i] for i in indexes][0:params.train_count]
    labels = [labels[i] for i in indexes][0:params.train_count]

    return np.array(features), np.array(labels)

def get_test_data():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_positive_features.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_negative_features.pkl', 'rb'))
    samples = positive_samples + negative_samples
    random.shuffle(samples)
    features, labels = [], []
    for sample in samples:
        if sample[0] is None or sample[1] is None:
            continue
        if params.with_fork:
            features.append(sample[0])
        else:
            features.append(sample[0][:15])
        labels.append(sample[1])

    indexes = list(range(len(features)))
    random.shuffle(indexes)
    features = [features[i] for i in indexes][0:params.test_count]
    labels = [labels[i] for i in indexes][0:params.test_count]

    return np.array(features), np.array(labels)

def get_test_data_topological():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_positive_features_topological.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_negative_features_topological.pkl', 'rb'))
    samples = positive_samples + negative_samples
    random.shuffle(samples)
    features, labels = [], []
    for sample in samples:
        if sample[0] is None or sample[1] is None:
            continue
        if params.use_subset:
            features.append(sample[0][0:12])
        else:
            features.append(sample[0])
        labels.append(sample[1])

    indexes = list(range(len(features)))
    random.shuffle(indexes)
    features = [features[i] for i in indexes][0:params.test_count]
    labels = [labels[i] for i in indexes][0:params.test_count]

    return np.array(features), np.array(labels)

def get_test_data_joint():
    positive_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_positive_features_joint.pkl', 'rb'))
    negative_samples = pickle.load(open('/home/social-sim/PycharmProjects/Information_Diffusion/data/test_data_negative_features_joint.pkl', 'rb'))
    samples = positive_samples + negative_samples
    random.shuffle(samples)
    features, labels = [], []
    for sample in samples:
        if sample[0] is None or sample[1] is None:
            continue
        features.append(sample[0][0:29])
        labels.append(sample[1])

    indexes = list(range(len(features)))
    random.shuffle(indexes)
    features = [features[i] for i in indexes][0:params.test_count]
    labels = [labels[i] for i in indexes][0:params.test_count]

    return np.array(features), np.array(labels)

def train(run_id):
    model = build_model(params.input_size)
    #optim = SGD(lr=params.lr, decay=params.decay, momentum=params.momentum, nesterov=True)
    optim = Adam(lr=params.lr, decay=params.decay)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    if params.use_topological:
        features, labels = get_train_data_topological()
    elif params.train_joint:
        features, labels = get_train_data_joint()
    else:
        features, labels = get_train_data()

    save_dir = './classifier_models/' + str(run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logs_dir = './logs/' + str(run_id)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    classifier_model_file = './classifier_models/' + str(run_id) + '/classifier.h5'
    filepath = os.path.join(save_dir, "classifier-model-improvement-{epoch:02d}-{val_acc:.2f}.h5")
    tensorboard = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True, write_images=False)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=False, save_best_only=True, mode='max')

    # define model
    model.fit(features, labels,
              batch_size=params.batch_size,
              epochs=params.epochs,
              validation_split=params.val_split,
              shuffle=True,
              callbacks=[tensorboard, checkpoint])

    model.save(classifier_model_file)

def test(saved_model_file):
    print(saved_model_file)
    model = load_model(saved_model_file)
    if params.use_topological:
        features, labels = get_test_data_topological()
    elif params.train_joint:
        features, labels = get_test_data_joint()
    else:
        features, labels = get_test_data()
    predictions = model.predict(features)
    predictions[predictions > 0.5] = 1
    predictions[predictions < 0.5] = 0
    p_r_f1_s = precision_recall_fscore_support(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    print("Neural Network Accuracy : ", accuracy)
    print("Neural Network Precision, Recall, F1-Score, Support: ", p_r_f1_s)
    return accuracy, p_r_f1_s

def train_random_forest():
    train_features, train_labels = get_train_data()
    test_features, test_labels = get_test_data()
    results_random_forest = {}
    results_random_forest['acc'] = []
    results_random_forest['p_r_f1_s'] = []
    for i in range(30):
        clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "Random Forest Accuracy : ", acc)
        print( "Random Forest Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_random_forest['acc'].append(acc)
        results_random_forest['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(10)
    pickle.dump(results_random_forest, open('results_random_forest.pkl', 'wb'))

def train_svm():
    results_linear = {}
    results_linear['acc'] = []
    results_linear['p_r_f1_s'] = []
    for i in range(30):
        train_features, train_labels = get_train_data()
        test_features, test_labels = get_test_data()
        clf = LinearSVC()
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "Linear SVM Accuracy : ", acc)
        print( "Linear SVM Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_linear['acc'].append(acc)
        results_linear['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(5)
    pickle.dump(results_linear, open('results_linear.pkl', 'wb'))

    results_rbf = {}
    results_rbf['acc'] = []
    results_rbf['p_r_f1_s'] = []
    for i in range(30):
        train_features, train_labels = get_train_data()
        test_features, test_labels = get_test_data()
        clf = SVC(kernel='rbf')
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "RBF SVM Accuracy : ", acc)
        print( "RBF SVM Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_rbf['acc'].append(acc)
        results_rbf['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(10)
    pickle.dump(results_rbf, open('results_rbf.pkl', 'wb'))

def train_knn():
    results_knn = {}
    results_knn['acc'] = []
    results_knn['p_r_f1_s'] = []
    for i in range(30):
        train_features, train_labels = get_train_data()
        test_features, test_labels = get_test_data()

        num_neighbors = random.randint(5, 50)
        weights = random.choice(['uniform', 'distance'])
        clf = neighbors.KNeighborsClassifier(num_neighbors, weights=weights)
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "K Nearest Neighbor : ", acc)
        print( "K Nearest Neighbor Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_knn['acc'].append(acc)
        results_knn['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(10)
    pickle.dump(results_knn, open('results_knn.pkl', 'wb'))

def train_nbc():
    results_nbc = {}
    results_nbc['acc'] = []
    results_nbc['p_r_f1_s'] = []
    for i in range(30):
        train_features, train_labels = get_train_data()
        test_features, test_labels = get_test_data()

        clf = MultinomialNB()
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "Naive Bayes Classifier : ", acc)
        print( "Naive Bayes Classifier Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_nbc['acc'].append(acc)
        results_nbc['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(10)
    pickle.dump(results_nbc, open('results_nbc.pkl', 'wb'))

def train_linear_model():
    results_linear_model = {}
    results_linear_model['acc'] = []
    results_linear_model['p_r_f1_s'] = []
    for i in range(30):
        train_features, train_labels = get_train_data()
        test_features, test_labels = get_test_data()

        clf = linear_model.SGDClassifier()
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "Linear Model Classifier : ", acc)
        print( "Linear Model Classifier Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_linear_model['acc'].append(acc)
        results_linear_model['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(10)
    pickle.dump(results_linear_model, open('results_linear_model.pkl', 'wb'))

def train_decisiontree_model():
    results_decisiontree_model = {}
    results_decisiontree_model['acc'] = []
    results_decisiontree_model['p_r_f1_s'] = []
    for i in range(30):
        train_features, train_labels = get_train_data()
        test_features, test_labels = get_test_data()

        clf = DecisionTreeClassifier()
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "DecisionTree Model Classifier : ", acc)
        print( "DecisionTree Model Classifier Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_decisiontree_model['acc'].append(acc)
        results_decisiontree_model['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(10)
    pickle.dump(results_decisiontree_model, open('results_decisiontree_model.pkl', 'wb'))

def train_extratree_model():
    results_extratree_model = {}
    results_extratree_model['acc'] = []
    results_extratree_model['p_r_f1_s'] = []
    for i in range(30):
        train_features, train_labels = get_train_data()
        test_features, test_labels = get_test_data()

        clf = ExtraTreeClassifier()
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "ExtraTree Model Classifier : ", acc)
        print( "ExtraTree Model Classifier Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_extratree_model['acc'].append(acc)
        results_extratree_model['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(10)
    pickle.dump(results_extratree_model, open('results_extratree_model.pkl', 'wb'))

def train_kmeans():
    results_kmeans = {}
    results_kmeans['acc'] = []
    results_kmeans['p_r_f1_s'] = []
    for i in range(30):
        train_features, train_labels = get_train_data()
        test_features, test_labels = get_test_data()

        predictions = KMeans(n_clusters=2, random_state=0).fit(test_features).labels_
        p_r_f1_s = precision_recall_fscore_support(test_labels, predictions)
        acc = accuracy_score(test_labels, predictions)
        print( "K Means : ", acc)
        print( "K Means Precision, Recall, F1-Score, Support: ", p_r_f1_s)
        results_kmeans['acc'].append(acc)
        results_kmeans['p_r_f1_s'].append(p_r_f1_s)
        time.sleep(10)
    pickle.dump(results_kmeans, open('results_kmeans.pkl', 'wb'))

if __name__ == '__main__':
    bs = [64, 128]
    lr = [1e-3, 1e-4]
    epochs = [100, 250]
    accuracies = []
    metrics = []
    for b in bs:
        for l in lr:
            for e in epochs:
                params.batch_size = b
                params.lr = l
                params.epochs = e
                #run_id = 'run_adam_bs' + str(b) + '_with_fork_lr' + str(l) + '_6layers_epochs_' + str(e)
                #run_id = 'run_adam_bs' + str(b) + '_lr' + str(l) + '_6layers_epochs_' + str(e)
                #run_id = 'run_topological_adam_bs' + str(b) + '_lr' + str(l) + '_6layers_epochs_' + str(e)
                #run_id = 'run_topological_adam_bs' + str(b) + '_lr' + str(l) + '_epochs_' + str(e)
                #run_id = 'run_2_topological_adam_bs' + str(b) + '_lr' + str(l) + '_epochs_' + str(e)
                #run_id = 'run_2_topological_adam_bs' + str(b) + '_lr' + str(l) + '_6layers_epochs_' + str(e)
                run_id = 'run_joint_adam_bs' + str(b) + '_lr' + str(l) + '_6layers_epochs_' + str(e)
                #train(run_id)
                acc, p_r_f1_s = test(os.path.join('./classifier_models', run_id, 'classifier.h5'))
                #accuracies.append(acc)
                #metrics.append(p_r_f1_s)
                #print(accuracies, metrics)

    #train('run_3_adam_bs64_with_fork_lr_1e-4_epochs_1000')
    #test('/home/social-sim/Info_Diff/classifier_models/run_3_adam_bs64_with_fork_lr_1e-4_epochs_1000/classifer.h5')

    #train_svm()
    #train_kmeans()
    #train_random_forest()
    #train_knn()
    #train_nbc()
    #train_linear_model()
    #train_decisiontree_model()
    #train_extratree_model()

    #test('/home/social-sim/Info_Diff/classifier_models/run_2_adam_bs128_lr_1e-3_epochs_1000_6layers/classifier-model-improvement-698-0.71.h5')
    #test('/home/social-sim/Info_Diff/classifier_models/run_2_adam_bs128_lr_1e-3_epochs_1000_6layers/classifier-model-improvement-921-0.70.h5')
    #test('/home/social-sim/Info_Diff/classifier_models/run_2_bs64_lr_1e-3_epochs_1000')
    #test('/home/social-sim/Info_Diff/classifier_models/run_2_adam_bs128_with_fork_lr_1e-3_epochs_1000_6layers/classifier-model-improvement-662-0.82.h5')
    #test('/home/social-sim/Info_Diff/classifier_models/run_2_adam_bs128_with_fork_lr_1e-3_epochs_500_6layers/classifier-model-improvement-480-0.80.h5')

    #test('/home/social-sim/Info_Diff/classifier_models/run_1_bs32_with_fork_lr_1e-3_epochs_500/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs8_shorter/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs32_lr_1e-3_epochs_500/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs8_with_fork_lr_1e-2_epochs_100/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs8_with_fork_lr_1e-3_epochs_100/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs32_with_fork_lr_1e-3_epochs_100/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs32_with_fork_lr_1e-3_epochs_100/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs32_with_fork_lr_1e-3_epochs_500/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs8_with_fork_lr_1e-3_epochs_500/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs64_with_fork_lr_1e-3_epochs_500/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs32_with_fork_lr_1e-3_epochs_1000/classifer.h5')
    # test('/home/social-sim/Info_Diff/classifier_models/run_1_bs32_with_fork_lr_1e-4_epochs_1000/classifer.h5')
