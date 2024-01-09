# Part 2
# October 22, 2023
# by Olivia Radcliiffe

import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import scikeras

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA    

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import layers

from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

#----------------------------------- Task 3- Data Analysis: (60%) ---------------------------------
# Notes:
# a) For the following subtasks, use the original dataset (without dimensionality reduction)
# b) You may use GridSearchCV from sklearn to try and determine the best hyperparameters. In
# all cases, you should show your exploration that resulted in the selected model parameters.
# c) Performance of all models should be reported on the test set by calculating accuracy,
# sensitivity, specificity, and confusion matrix.

# Helper function for questions 2-6 for getting the model performance metrics
# calculating accuracy,sensitivity, specificity, and confusion matrix.
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    confMat = confusion_matrix(y_test, y_predicted)
    tp = confMat[1, 1]  # True positives
    fn = confMat[1, 0]  # False negatives
    sensitivity = tp / (tp + fn)
    tn = confMat[0, 0]  # True negatives
    fp = confMat[0, 1]  # False positives
    specificity = tn / (tn + fp)
    
    return accuracy, precision, confMat, sensitivity, specificity


# 1. Establishs a simple baseline model, by assigning the label of the majority class to all data
#  and calculating the accuracy on test set. Other models should not perform worse than this.
def baselineModel(train_labels, test_labels):
    #  calculate the majority class
    majority_class = np.bincount(train_labels).argmax()

    # assign the majority class to all test samples
    baseline_model_pred = np.full_like(test_labels, fill_value=majority_class)

    return baseline_model_pred


# 2/3. Uses Logistic regression from sklearn predict the class (cancer type in this case)
#  and reports its performance by printing accuracy, sensitivity, specificity,
#  and confusion matrix.
def question3_logReg(train_data, train_labels, test_data): 
    # create LogisticRegression instance
    logReg = LogisticRegression(random_state=0, C=10, penalty='l1', solver='liblinear')

    def logReg_explore():
        param_grid = dict()
        param_grid['solver'] = ['newton-cg', 'liblinear']
        param_grid['penalty'] = ['l1', 'l2']
        param_grid['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

        # define search
        search = GridSearchCV(logReg, param_grid, scoring='accuracy', cv=5, verbose=0)
        # execute search
        result = search.fit(train_data, train_labels)

        logReg = result.best_estimator_

        return logReg
    
    # UNCOMMENT FOR GRIDSEARCH AND COMMENT BELOW
    #logReg = logReg_explore()
    logReg.fit(train_data, train_labels)

    # test model
    logReg_pred = logReg.predict(test_data)

    return logReg_pred


# 4. Uses a decision tree from sklearn to predict the class (cancer type in this case) and 
# reports its performance as in subtask 2/3.
def question4_decTree(train_data, train_labels, test_data):
    decision_tree = tree.DecisionTreeClassifier(random_state=5, criterion='gini', max_depth=2, min_samples_split=2)

    def decTree_explore():
        param_grid = dict()
        param_grid['min_samples_split'] = range(1,5)
        param_grid['criterion'] = ['gini', 'entropy']
        param_grid['max_depth'] = [2,4,6,8,10,12]
        
        # define search
        search = GridSearchCV(decision_tree, param_grid, scoring='accuracy', cv=5, verbose=0)
        # execute search
        result = search.fit(train_data, train_labels)
        
        decision_tree = result.best_estimator_

        return decision_tree
    
    # UNCOMMENT FOR GRIDSEARCH AND COMMENT BELOW
    #decision_tree = decTree_explore()
    decision_tree.fit(train_data, train_labels)

    #test model
    decTree_pred = decision_tree.predict(test_data)

    return decTree_pred


# 5. Uses a random forest from sklearn to predict the cancer type and report its performance
# as in subtask 2.
def question5_randForest(train_data, train_labels, test_data):

    def randForest_explore():

        randForest_model = RandomForestClassifier(random_state=7)

        param_grid = dict()
        param_grid['criterion'] = ['gini', 'entropy', 'log_loss']
        param_grid['max_depth'] = [1,2,4,6,8,10,None]
        param_grid['max_features'] = ['sqrt', 'log2']
        param_grid['max_leaf_nodes'] = range(1,5)

        
        # define search
        search = GridSearchCV(randForest_model, param_grid, scoring='accuracy', cv=5, verbose=0)
        # execute search
        result = search.fit(train_data, train_labels)

        randForest_model = result.best_estimator_

        return randForest_model
    
    # UNCOMMENT FOR GRIDSEARCH AND COMMENT BELOW
    #randForest_model = randForest_explore(randForest_model)
    randForest_model = RandomForestClassifier(random_state=7, criterion='entropy', max_depth=1, max_features='sqrt', max_leaf_nodes=2)
    randForest_model.fit(train_data, train_labels)

    # test model
    randForest_pred = randForest_model.predict(test_data)

    return randForest_pred




# 6. Build an ANN (using tensorflow-keras) to predict the cancer type and report its performance as in subtask 2.
# Choose an appropriate architecture. Explain how you have decided to choose the parameters such as learning rate,
# batch size, number of epochs, size of the hidden layer, number of hidden layers, etc. As the epochs go by, we
# expect that its error on the training set naturally goes down. But we are not actually sure that overfitting
# is happening or not. One thing we can do, is to further split the training set into train and validation and
# monitor the validation loss as we do for the train loss. If after a while, the validation error stops 
# decreasing, this indicates that the model has started to overfit the training data. With Early Stopping, 
# you just stop training as soon as the validation error reaches the minimum. Try to add Early stopping using 
# keras callbacks to your model.
def question6_buildANN(train_data, train_labels, test_data, test_labels):

    earlyStopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='auto',
                baseline=None,
                restore_best_weights=True,
                verbose= 0)

    # select hyperparameters
    lossFunction = 'binary_crossentropy'
    metrics =["accuracy"]
    optimizer = 'Adam'
    batch_size = 16
    epochs = 10
    
    # define the model
    def create_model():
        model = keras.Sequential()
        model.add(layers.Dense(64, input_shape=(train_data.shape[1],), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation = 'sigmoid'))

        return model
    
    def create_explore_model():
        model = keras.Sequential()
        model.add(layers.Dense(64, input_shape=(train_data.shape[1],), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation = 'sigmoid'))

        model.compile(optimizer=optimizer, loss=lossFunction, metrics=metrics)

        return model


    def ann_explore():
        model = KerasClassifier(model=create_explore_model,callbacks=earlyStopping, validation_split=0.3, loss=lossFunction, metrics=metrics, verbose=0, random_state=2)

        param_grid = dict()
        param_grid['batch_size'] = [8, 16, 32]
        param_grid['epochs'] = [10, 20, 30]
        param_grid['optimizer'] = ['SGD', 'RMSprop', 'Adam']

        # define search
        search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
        # execute search
        result = search.fit(train_data, train_labels, verbose=0)

        model = result.best_estimator_

        return model, result.best_params_
    
    
    # UNCOMMENT FOR GRIDSEARCH AND COMMENT BELOW
    #model, best_params = ann_explore()    
    model = KerasClassifier(model=create_model,callbacks=earlyStopping, validation_split=0.3, loss=lossFunction, metrics=metrics, verbose=0, random_state=2)
    model.fit(train_data, np.array(train_labels), epochs=epochs, batch_size=batch_size, verbose=0)

    # test model
    ann_pred = model.predict(test_data)
  
    return ann_pred



# 7. Each time you train a network, there is a set of new initialization of parameters, 
# (e.g. weights in neural networks) and so the performance differs. How can you make 
# sure that your results are reproducible?

# See Report PDF


# 8. Which of the above results in the best performance? Why do you think that is?

# See Report PDF


# helper function to print performance metrics
def print_stats(modelName, test_labels, y_pred):

    # calculate performance metrics for the logistic regression model
    print("\n%s Performance Metrics" %modelName)

    accuracy, precision, confMat, sensitivity, specificity = get_metrics(np.array(test_labels), y_pred)
    print("accuracy = %.3f \nprecision = %.3f \nsensitivity = %.3f \nspecificity = %.3f \nconfusion matrix: \n%s" % (accuracy, precision, sensitivity, specificity, confMat))


def main():
    # LABEL DATASET
    # importing the label dataset
    directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    labels_df = pd.read_csv(directory + '/data/actual.csv')

    # extract classes
    values = np.array(labels_df['cancer'].ravel())

    # endode classes to integers
    labelEncoder = LabelEncoder()
    integer_labels = labelEncoder.fit_transform(values)


    # TRAIN DATASET

    # import train data file
    train_data_df = pd.read_csv(directory + '/data/data_set_ALL_AML_train.csv')
    cleaned_train_data_df = train_data_df.copy()

    # remove ‘Call’ columns from train data file
    for col in train_data_df.columns:
        if 'call' in col:
            cleaned_train_data_df = cleaned_train_data_df.drop(columns=col)


    # TEST DATASET

    # import test data file
    test_data_df = pd.read_csv(directory + '/data/data_set_ALL_AML_independent.csv')
    cleaned_test_data_df = test_data_df.copy()

    # remove ‘Call’ columns from test data file
    for col in test_data_df.columns:
        if 'call' in col:
            cleaned_test_data_df = cleaned_test_data_df.drop(columns=col)


    # ASSOCIATE LABELS

    # initialize  train and test data label lists
    train_labels = []
    test_labels = []

    # find integer labels for train data and add to list
    for patientNum in cleaned_train_data_df.columns[2:]:
        train_labels.append(integer_labels[int(patientNum)-1])

    # find integer labels for test data and add to list
    for col in cleaned_test_data_df.columns[2:]:
        test_labels.append(integer_labels[int(col)-1])


    # STANDARDIZE DATA

    # standardize instance
    scaler = StandardScaler()

    # standardizing train data
    standardized_train_data = scaler.fit_transform((cleaned_train_data_df.iloc[:,2:]).T)
    # Convert the standardized data back to a DataFrame     
    standardized_train_df = pd.DataFrame(standardized_train_data, columns=cleaned_train_data_df.index)

    # standardizing test data
    standardized_test_data = scaler.fit_transform((cleaned_test_data_df.iloc[:,2:]).T)
    # Convert the standardized data back to a DataFrame
    standardized_test_df = pd.DataFrame(standardized_test_data, columns=cleaned_test_data_df.index)


    # -------------------------------- Task 2- Dimensionality Reduction: --------------------------
    pca = PCA(n_components=0.9)
    train_data_reduced = pca.fit_transform(standardized_train_df)

    # reduce test set
    test_data_reduced = pca.transform(standardized_test_df)


    #--------------------------------- Task 3 - Data Analysis: (60%)  ------------------------------
    baseline_pred = baselineModel(train_labels, test_labels)
    # calculate performance metrics for the baseline model
    baseAccuracy, basePrecision, baseConfMat, baseSensitivity, baseSpecificity = get_metrics(test_labels, baseline_pred)

    # Logistic regression 
    logReg_pred = question3_logReg(standardized_train_df, train_labels, standardized_test_df)
    # Decision tree 
    decTree_pred = question4_decTree(standardized_train_df, train_labels, standardized_test_df)
    # Random Forest classifier
    randForest_pred = question5_randForest(standardized_train_df, train_labels, standardized_test_df)
    # ANN
    ann_pred = question6_buildANN(standardized_train_df, train_labels, standardized_test_df, test_labels)
    

    # 9. Repeat parts 2-8 using the dimensionally reduced dataset (Task 2) and compare the performance
    # of all models with the original dataset. What are your conclusions?

    # Logistic regression 
    reduced_logReg_pred = question3_logReg(train_data_reduced, train_labels, test_data_reduced)
    # Decision tree 
    reduced_decTree_pred = question4_decTree(train_data_reduced, train_labels, test_data_reduced)
    # Random Forest classifier
    reduced_randForest_pred = question5_randForest(train_data_reduced, train_labels, test_data_reduced)
    # ANN
    reduced_ann_pred = question6_buildANN(train_data_reduced, train_labels, test_data_reduced, test_labels)

    # PRINT METRICS
    print("\nBaseline Model Performance Metrics")
    print("accuracy = %.3f \nprecision = %.3f \nsensitivity = %.3f \nspecificity = %.3f \nconfusion matrix: \n%s" % (baseAccuracy, basePrecision, baseSensitivity, baseSpecificity, baseConfMat))
    print_stats("Logistic Regression", test_labels, logReg_pred)
    print_stats("Decision Tree", test_labels, decTree_pred)
    print_stats("Random Forest", test_labels, randForest_pred)
    print_stats("ANN", test_labels, ann_pred)

    
    print("\n9. DIM REDUCTION - PERFORMANCE METRICS")
    print_stats("Logistic Regression - Reduced", test_labels, reduced_logReg_pred)
    print_stats("Decision Tree - Reduced", test_labels, reduced_decTree_pred)
    print_stats("Random Forest - Reduced", test_labels, reduced_randForest_pred)
    print_stats("ANN - Reduced", test_labels, reduced_ann_pred)


if __name__ == "__main__":
    main()