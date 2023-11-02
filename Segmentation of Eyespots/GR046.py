# Machine Learning 2022/23 3rd Lab

# Afonso Alemão 96135
# Rui Daniel 96317

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction import image
import math   
from imblearn.over_sampling import RandomOverSampler 


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

def MLP_kfolds(size_network, kk, layer_increasing, hidden_layer_start, balance, mode_imbalance, number_classes):
    if size_network < 1:
        size_network = 1     
    class_weight = {0: 1, 1: 1, 2: 1}
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')

    # Convert train labels to one-hot encoding
    y_train_classification = tf.keras.utils.to_categorical(y_train, num_classes=number_classes)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    balanced_accuracy_folds = []
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x_train, y_train_classification):
        x_val = x_train[test]
        y_val_classification = y_train_classification[test]
        x_trainn = x_train[train]
        y_train_classificationn = y_train_classification[train]
        
        print(f"Fold no = {fold_no}")
        fold_no += 1
        
        x_new_val, x_val, y_new_val_classification, y_val_classification = train_test_split(x_val, y_val_classification, test_size=0.5, random_state=1234)

        y_train_classificationn_1D = np.argmax(y_train_classificationn, axis=1)
        if balance: 
            if mode_imbalance == 1: # Oversampling with SMOTE
                sm = SMOTE(random_state=42)
                x_trainn, y_train_classificationn_1D = sm.fit_resample(x_trainn, y_train_classificationn_1D)
            elif mode_imbalance == 5: # Oversampling with RandomOverSampler
                oversample = RandomOverSampler()
                x_trainn, y_train_classificationn_1D = oversample.fit_resample(x_trainn, y_train_classificationn_1D)
            elif mode_imbalance == 2: # Undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_train_classificationn_1D = undersample.fit_resample(x_trainn, y_train_classificationn_1D)
            elif mode_imbalance == 3: # class weight 
                class1_pos = np.count_nonzero(y_train_classificationn_1D == 1)
                class2_pos = np.count_nonzero(y_train_classificationn_1D == 2)
                total = y_train_classificationn_1D.shape[0]
                class0_pos = total - class1_pos - class2_pos
                weight_for_0 = (1 / class0_pos) * (total / 3.0)
                weight_for_1 = (1 / class1_pos) * (total / 3.0)
                weight_for_2 = (1 / class2_pos) * (total / 3.0)
                class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
                
                print('Weight for class 0: {:.2f}'.format(weight_for_0))
                print('Weight for class 1: {:.2f}'.format(weight_for_1))
                print('Weight for class 2: {:.2f}'.format(weight_for_2))
            else: # Oversampling with our implementation
                x_trainn, y_train_classificationn_1D = Oversampling(x_trainn, y_train_classificationn_1D)
                
        y_train_classificationn = tf.keras.utils.to_categorical(y_train_classificationn_1D, num_classes=number_classes)

        # Sequential model. Start by adding a flatten layer to convert the 3D images to 1D. 
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(75,1), name="initial"))
        
        # Add hidden layers to MLP.
        # Use ’relu’ as activation function for all neurons in the hidden layers.
        layer_size = hidden_layer_start
        model.add(layers.Rescaling(1./255))
        for i in range(size_network):    
            namee = "hidden_" + str(i + 1)
            model.add(layers.Dense(layer_size, activation="relu", name=namee))
            if layer_increasing:
                layer_size = layer_size * 2
        
        # End the network with a softmax layer. This is a dense layer that will return an array
        # of 2 probability scores, one per class, summing to 1.
        model.add(layers.Dense(number_classes, activation="softmax", name="final"))

        # Get the summary of network to check it is correct.
        # model.build(input_shape = x_trainn.shape)
        # model.summary()

        # Create an Early stopping monitor that will stop training when the validation loss is not improving
        callback = EarlyStopping(patience=10, restore_best_weights=True)

        # Fit the MLP to training and validation data using ’categorical_crossentropy’ as the loss function, 
        # a batch size of 200 (mini-batch) and Adam as the optimizer (learning rate=0.001, clipnorm=1). 
        # Choose, as stopping criterion, the number of epochs reaching 200. 
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = [tf.keras.metrics.Recall(class_id = 0), 
                                                                             tf.keras.metrics.Recall(class_id = 1),
                                                                             tf.keras.metrics.Recall(class_id = 2)])         
        model.fit(class_weight=class_weight, verbose=0, x=x_trainn, y=y_train_classificationn, batch_size=32, epochs=200, callbacks=[callback], validation_data=(x_val, y_val_classification))
        
        # Evaluate performance 
        y_new_pred_classification = model.predict(x_new_val)
        
        y_new_pred = np.argmax(y_new_pred_classification, axis=1)
        y_new_val = np.argmax(y_new_val_classification, axis=1)
        
        # print(f'y_val.shape = {y_val.shape}')
        # print(f'y_val_classification.shape = {y_val_classification.shape}')
        # print(f'y_pred = {y_pred.shape}')
        # print(f'y_pred_classification = {y_pred_classification.shape}')
        # print(f'y_train_classification.shape = {y_train_classification.shape}')
        
        conf_matrix_fold = confusion_matrix(y_new_val, y_new_pred)
        balanced_accuracy_fold = balanced_accuracy_score(y_new_val, y_new_pred)
        
        print(f"conf_matrix: {conf_matrix_fold}")
        print(f"balanced_accuracy_score: {balanced_accuracy_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        balanced_accuracy_folds.append(balanced_accuracy_fold)
    
    balanced_accuracy = np.average(balanced_accuracy_folds)
    conf_matrix = np.empty([number_classes, number_classes])
    for k in range(len(conf_matrix_folds)):
        for i in range(number_classes):
            for j in range(number_classes):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    return [balanced_accuracy, conf_matrix]


def MLP_testing(max_number_layers, kk, layer_increasing, hidden_layer_start, balance, mode_imbalance, number_classes):
    print(f"Using kfolds = {kk}")
    balanced_accuracy = []
    confusion_matrix = []
    for i in range(max_number_layers):
        (balanced_accuracy_i, confusion_matrix_i) = MLP_kfolds(i + 1, kk, layer_increasing, hidden_layer_start, balance, mode_imbalance, number_classes)
        if (i == 0):
            print(f"For {i + 1} layer:")
        else:
            print(f"For {i + 1} layers:")
        print(f"Average conf_matrix: {confusion_matrix_i}")
        print(f"Average balanced_accuracy_score: {balanced_accuracy_i}")
        balanced_accuracy.append(balanced_accuracy_i)
        confusion_matrix.append(confusion_matrix_i)
    best_num_layer = (balanced_accuracy.index(max(balanced_accuracy)) + 1)
    print(f'Best case: Number of layers = {best_num_layer} with balanced_accuracy = {max(balanced_accuracy)}')
    
def MLP(size_network, layer_increasing, hidden_layer_start, balance, mode_imbalance, number_classes):
    class_weight = {0: 1, 1: 1, 2: 1}
    if size_network < 1:
        size_network = 1     
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')
    x_test = np.load('data/Xtest_Classification2.npy')

    # Convert train labels to one-hot encoding
    y_train_classification = tf.keras.utils.to_categorical(y_train, num_classes=number_classes)
    
    x_train, x_val, y_train_classification, y_val_classification = train_test_split(x_train, y_train_classification, test_size=0.3, random_state=1234)
    x_new_val, x_val, y_new_val_classification, y_val_classification = train_test_split(x_val, y_val_classification, test_size=0.5, random_state=1234)
        
    y_train_classification_1D = np.argmax(y_train_classification, axis=1)
    if balance: 
        if mode_imbalance == 1: # Oversampling with SMOTE
            sm = SMOTE(random_state=42)
            x_train, y_train_classification_1D = sm.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 5: # Oversampling with RandomOverSampler
            oversample = RandomOverSampler()
            x_train, y_train_classification_1D = oversample.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 2: # Undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train_classification_1D = undersample.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 3: # class weight 
            class1_pos = np.count_nonzero(y_train_classification_1D == 1)
            class2_pos = np.count_nonzero(y_train_classification_1D == 2)
            total = y_train_classification_1D.shape[0]
            class0_pos = total - class1_pos - class2_pos
            weight_for_0 = (1 / class0_pos) * (total / 3.0)
            weight_for_1 = (1 / class1_pos) * (total / 3.0)
            weight_for_2 = (1 / class2_pos) * (total / 3.0)
            class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
            
            print('Weight for class 0: {:.2f}'.format(weight_for_0))
            print('Weight for class 1: {:.2f}'.format(weight_for_1))
            print('Weight for class 2: {:.2f}'.format(weight_for_2))
        else: # Oversampling with our implementation
            x_train, y_train_classification_1D = Oversampling(x_train, y_train_classification_1D)
            
    y_train_classification = tf.keras.utils.to_categorical(y_train_classification_1D, num_classes=number_classes)
    
    # Sequential model. Start by adding a flatten layer to convert the 3D images to 1D. 
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(75,1), name="initial"))
    
    # Add hidden layers to MLP.
    # Use ’relu’ as activation function for all neurons in the hidden layers.
    layer_size = hidden_layer_start
    model.add(layers.Rescaling(1./255))
    for i in range(size_network):    
        namee = "hidden_" + str(i + 1)
        model.add(layers.Dense(layer_size, activation="relu", name=namee))
        if layer_increasing:
            layer_size = layer_size * 2
    
    # End the network with a softmax layer. This is a dense layer that will return an array
    # of 2 probability scores, one per class, summing to 1.
    model.add(layers.Dense(number_classes, activation="softmax", name="final"))

    # Get the summary of network to check it is correct.
    model.build(input_shape = x_train.shape)
    model.summary()

    # Create an Early stopping monitor that will stop training when the validation loss is not improving
    callback = EarlyStopping(patience=10, restore_best_weights=True)

    # Fit the MLP to training and validation data using ’categorical_crossentropy’ as the loss function, 
    # a batch size of 200 (mini-batch) and Adam as the optimizer (learning rate=0.001, clipnorm=1). 
    # Choose, as stopping criterion, the number of epochs reaching 200. 
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = [tf.keras.metrics.Recall(class_id = 0), 
                                                                             tf.keras.metrics.Recall(class_id = 1),
                                                                             tf.keras.metrics.Recall(class_id = 2)])        

    model.fit(class_weight=class_weight, x=x_train, y=y_train_classification, batch_size=32, epochs=200, callbacks=[callback], validation_data=(x_val, y_val_classification))
    
    # Evaluate performance 
    y_val_pred_classification = model.predict(x_val)
    y_new_val_pred_classification = model.predict(x_new_val)
    y_test_pred_classification = model.predict(x_test)

    y_pred_val = np.argmax(y_val_pred_classification, axis=1)
    y_val = np.argmax(y_val_classification, axis=1)
    y_pred_new_val = np.argmax(y_new_val_pred_classification, axis=1)
    y_new_val = np.argmax(y_new_val_classification, axis=1)
    y_pred_test = np.argmax(y_test_pred_classification, axis=1)
    
    print(f'y_pred_train = {y_pred_val.shape}')
    print(f'y_train_pred_classification = {y_val_pred_classification.shape}')
    print(f'y_pred_test = {y_pred_test.shape}')
    print(f'y_test_pred_classification = {y_test_pred_classification.shape}')
    
    conf_matrix_irr = confusion_matrix(y_val, y_pred_val)
    balanced_accuracy_irr = balanced_accuracy_score(y_val, y_pred_val)
    
    print(f"Irrelevant - conf_matrix of validation data: {conf_matrix_irr}")
    print(f"Irrelevant - balanced_accuracy_score of validation data: {balanced_accuracy_irr}")
    
    conf_matrix = confusion_matrix(y_new_val, y_pred_new_val)
    balanced_accuracy = balanced_accuracy_score(y_new_val, y_pred_new_val)
    
    print(f"conf_matrix of new validation data: {conf_matrix}")
    print(f"balanced_accuracy_score of new validation data: {balanced_accuracy}")
    
    np.save('data/ytest_Classification2', y_pred_test)
    

def KNeighborsClassifier_kfolds(neighbors, kk, balance, mode_imbalance, number_classes):
    if neighbors < 1:
        neighbors = 5    
    x_train = np.load('data/Xtrain_Classification2.npy')
    x_train = x_train / 255 
    y_train = np.load('data/Ytrain_Classification2.npy')

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    balanced_accuracy_folds = []
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x_train, y_train):
        x_val = x_train[test]
        y_val = y_train[test]
        x_trainn = x_train[train]
        y_trainn = y_train[train]
        
        print(f"Fold no = {fold_no}")
        fold_no += 1
        
        if balance: 
            if mode_imbalance == 1: # Oversampling with SMOTE
                sm = SMOTE(random_state=42)
                x_trainn, y_trainn = sm.fit_resample(x_trainn, y_trainn)
            elif mode_imbalance == 5: # Oversampling with RandomOverSampler
                oversample = RandomOverSampler()
                x_trainn, y_trainn = oversample.fit_resample(x_trainn, y_trainn)
            elif mode_imbalance == 5: # Oversampling with RandomOverSampler
                oversample = RandomOverSampler()
                x_trainn, y_trainn = oversample.fit_resample(x_trainn, y_trainn)
            elif mode_imbalance == 2: # Undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_trainn = undersample.fit_resample(x_trainn, y_trainn)
            else: # Oversampling with our implementation
                x_trainn, y_trainn = Oversampling(x_trainn, y_trainn)
        
        neigh = KNeighborsClassifier(n_neighbors=neighbors)
        neigh.fit(x_trainn, y_trainn)
        y_pred_val = neigh.predict(x_val)
        
        # print(f'y_val.shape = {y_val.shape}')
        # print(f'y_pred_val.shape = {y_pred_val.shape}')
        # print(f'y_trainn.shape = {y_trainn.shape}')
        
        conf_matrix_fold = confusion_matrix(y_val, y_pred_val)
        balanced_accuracy_fold = balanced_accuracy_score(y_val, y_pred_val)
        
        # print(f"conf_matrix: {conf_matrix_fold}")
        # print(f"balanced_accuracy_score: {balanced_accuracy_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        balanced_accuracy_folds.append(balanced_accuracy_fold)
    
    balanced_accuracy = np.average(balanced_accuracy_folds)
    conf_matrix = np.empty([number_classes, number_classes])
    for k in range(len(conf_matrix_folds)):
        for i in range(number_classes):
            for j in range(number_classes):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    return [balanced_accuracy, conf_matrix]


def KNeighborsClassifier_testing(max_number_neighbors, kk, balance, mode_imbalance, number_classes):
    print(f"Using kfolds = {kk}")
    balanced_accuracy = []
    confusion_matrix = []
    num_neighbors = []
    for i in range(max_number_neighbors):
        if (i + 1) % 2 == 1:
            (balanced_accuracy_i, confusion_matrix_i) = KNeighborsClassifier_kfolds(i + 1, kk, balance, mode_imbalance, number_classes)
            if (i == 0):
                print(f"For {i + 1} neighbor:")
            else:
                print(f"For {i + 1} neighbors:")
            # print(f"Average conf_matrix: {confusion_matrix_i}")
            print(f"Average balanced_accuracy_score: {balanced_accuracy_i}")
            balanced_accuracy.append(balanced_accuracy_i)
            confusion_matrix.append(confusion_matrix_i)
            num_neighbors.append(i + 1)
    best_num_neighbors = num_neighbors[balanced_accuracy.index(max(balanced_accuracy))]
    print(f'Best case: Number of neighbors = {best_num_neighbors} with balanced_accuracy = {max(balanced_accuracy)}')

def KNeighborsclassifier(neighbors, balance, mode_imbalance):
    if neighbors < 1:
        neighbors = 5    
    x_train = np.load('data/Xtrain_Classification2.npy') 
    x_train = x_train / 255 
    y_train = np.load('data/Ytrain_Classification2.npy')
    x_test = np.load('data/Xtest_Classification2.npy')
    x_test = x_test / 255 

    if balance: 
        if mode_imbalance == 1: # Oversampling with SMOTE
            sm = SMOTE(random_state=42)
            x_train, y_train = sm.fit_resample(x_train, y_train)
        elif mode_imbalance == 5: # Oversampling with RandomOverSampler
            oversample = RandomOverSampler()
            x_train, y_train = oversample.fit_resample(x_train, y_train)
        elif mode_imbalance == 2: # Undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train = undersample.fit_resample(x_train, y_train)
        else: # Oversampling with our implementation
            x_train, y_train = Oversampling(x_train, y_train)
                
    neigh = KNeighborsClassifier(n_neighbors=neighbors)
    neigh.fit(x_train, y_train)
    y_pred_train = neigh.predict(x_train)
    y_pred_test = neigh.predict(x_test)
    
    print(f'y_train.shape = {y_train.shape}')
    print(f'y_pred_train.shape = {y_pred_train.shape}')
    print(f'y_pred_test.shape = {y_pred_test.shape}')
    print(f'x_test.shape = {x_test.shape}')
    
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    balanced_accuracy = balanced_accuracy_score(y_train, y_pred_train)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - balanced_accuracy_score of train data: {balanced_accuracy}")
    
    np.save('data/ytest_Classification2', y_pred_test)

def NaiveBayes_kfolds(kk, mode, balance, mode_imbalance, number_classes):
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')
    x_train = x_train / 255 
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    balanced_accuracy_folds = []
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x_train, y_train):
        x_val = x_train[test]
        y_val = y_train[test]
        x_trainn = x_train[train]
        y_trainn = y_train[train]
        
        print(f"Fold no = {fold_no}")
        fold_no += 1
        
        if balance: 
            if mode_imbalance == 1: # Oversampling with SMOTE
                sm = SMOTE(random_state=42)
                x_trainn, y_trainn = sm.fit_resample(x_trainn, y_trainn)
            elif mode_imbalance == 5: # Oversampling with RandomOverSampler
                oversample = RandomOverSampler()
                x_trainn, y_trainn = oversample.fit_resample(x_trainn, y_trainn)
            elif mode_imbalance == 2: # Undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_trainn = undersample.fit_resample(x_trainn, y_trainn)
            else: # Oversampling with our implementation
                x_trainn, y_trainn = Oversampling(x_trainn, y_trainn)
                
        if mode == 1:
            gnb = GaussianNB().fit(x_trainn, y_trainn)
        elif mode == 2:
            gnb = MultinomialNB().fit(x_trainn, y_trainn)
        elif mode == 3:
            gnb = ComplementNB().fit(x_trainn, y_trainn)
        else:
            gnb = BernoulliNB().fit(x_trainn, y_trainn)
        
        y_pred_val = gnb.predict(x_val)
        
        # print(f'y_val.shape = {y_val.shape}')
        # print(f'y_pred_val.shape = {y_pred_val.shape}')
        # print(f'y_trainn.shape = {y_trainn.shape}')
        
        conf_matrix_fold = confusion_matrix(y_val, y_pred_val)
        balanced_accuracy_fold = balanced_accuracy_score(y_val, y_pred_val)
        
        # print(f"conf_matrix: {conf_matrix_fold}")
        # print(f"balanced_accuracy_score: {balanced_accuracy_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        balanced_accuracy_folds.append(balanced_accuracy_fold)
    
    balanced_accuracy = np.average(balanced_accuracy_folds)
    conf_matrix = np.empty([number_classes, number_classes])
    for k in range(len(conf_matrix_folds)):
        for i in range(number_classes):
            for j in range(number_classes):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    
    print(f"for k = {kk} folds")
    print(f"conf_matrix: {conf_matrix}")
    print(f"balanced_accuracy_score: {balanced_accuracy}")

def NaiveBayesclassifier(mode, balance, mode_imbalance, number_classes):
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')
    x_test = np.load('data/Xtest_Classification2.npy')
    x_train = x_train / 255 
    x_test = x_test / 255 
    
    if balance: 
        if mode_imbalance == 1: # Oversampling with SMOTE
            sm = SMOTE(random_state=42)
            x_train, y_train = sm.fit_resample(x_train, y_train)
        elif mode_imbalance == 5: # Oversampling with RandomOverSampler
            oversample = RandomOverSampler()
            x_train, y_train = oversample.fit_resample(x_train, y_train)
        elif mode_imbalance == 2: # Undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train = undersample.fit_resample(x_train, y_train)
        else: # Oversampling with our implementation
            x_train, y_train = Oversampling(x_train, y_train)
                
    if mode == 1:
        gnb = GaussianNB().fit(x_train, y_train)
    elif mode == 2:
        gnb = MultinomialNB().fit(x_train, y_train)
    elif mode == 3:
        gnb = ComplementNB().fit(x_train, y_train)
    else:
        gnb = BernoulliNB().fit(x_train, y_train)
        
    y_pred_train = gnb.predict(x_train)
    y_pred_test = gnb.predict(x_test)
    
    print(f'y_train.shape = {y_train.shape}')
    print(f'y_pred_train.shape = {y_pred_train.shape}')
    print(f'y_pred_test.shape = {y_pred_test.shape}')
    
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    balanced_accuracy = balanced_accuracy_score(y_train, y_pred_train)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - balanced_accuracy_score of train data: {balanced_accuracy}")

    np.save('data/ytest_Classification2', y_pred_test)

def CNN_kfolds(kk, balance, mode_imbalance, num_iterations, number_classes):  
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')
    class_weight = {0: 1, 1: 1, 2: 1}
    
    # Convert train labels to one-hot encoding
    y_train_classification = tf.keras.utils.to_categorical(y_train, num_classes=number_classes)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    balanced_accuracy_folds = []
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for kkk in range(num_iterations):
        for train, test in kfold.split(x_train, y_train_classification):
            x_val = x_train[test]
            y_val_classification = y_train_classification[test]
            x_trainn = x_train[train]
            y_train_classificationn = y_train_classification[train]
            
            x_new_val, x_val, y_new_val_classification, y_val_classification = train_test_split(x_val, y_val_classification, test_size=0.5, random_state=1234)

            print(f"Fold no = {fold_no}")
            fold_no += 1

            y_train_classificationn_1D = np.argmax(y_train_classificationn, axis=1)
            if balance: 
                if mode_imbalance == 1: # Oversampling with SMOTE with SMOTE
                    sm = SMOTE(random_state=42)
                    x_trainn, y_train_classificationn_1D = sm.fit_resample(x_trainn, y_train_classificationn_1D)
                elif mode_imbalance == 5: # Oversampling with RandomOverSampler
                    oversample = RandomOverSampler()
                    x_trainn, y_train_classificationn_1D = oversample.fit_resample(x_trainn, y_train_classificationn_1D)
                elif mode_imbalance == 2: # Undersampling 
                    undersample = RandomUnderSampler(sampling_strategy='majority')
                    x_trainn, y_train_classificationn_1D = undersample.fit_resample(x_trainn, y_train_classificationn_1D)
                elif mode_imbalance == 3: # class weight
                    class1_pos = np.count_nonzero(y_train_classificationn_1D == 1)
                    class2_pos = np.count_nonzero(y_train_classificationn_1D == 2)
                    total = y_train_classificationn_1D.shape[0]
                    class0_pos = total - class1_pos - class2_pos
                    weight_for_0 = (1 / class0_pos) * (total / 3.0)
                    weight_for_1 = (1 / class1_pos) * (total / 3.0)
                    weight_for_2 = (1 / class2_pos) * (total / 3.0)
                    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
                    
                    print('Weight for class 0: {:.2f}'.format(weight_for_0))
                    print('Weight for class 1: {:.2f}'.format(weight_for_1))
                    print('Weight for class 2: {:.2f}'.format(weight_for_2))
                else: # Oversampling with our implementation
                    x_trainn, y_train_classificationn_1D = Oversampling(x_trainn, y_train_classificationn_1D)
                    
            y_train_classificationn = tf.keras.utils.to_categorical(y_train_classificationn_1D, num_classes=number_classes)

            x_trainn = transformXimage(x_trainn, x_trainn.shape[0], 5)
            x_val = transformXimage(x_val, x_val.shape[0], 5)
            x_new_val = transformXimage(x_new_val, x_new_val.shape[0], 5)

            model_cnn = tf.keras.Sequential()
            data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip("horizontal_and_vertical",
                                    input_shape=(x_trainn.shape[1],
                                                x_trainn.shape[2],
                                                x_trainn.shape[3])),
                    layers.RandomRotation(0.1),
                ]
            )
            model_cnn.add(data_augmentation)
            model_cnn.add(layers.Rescaling(1./255, input_shape=(5,5,3)))
            model_cnn.add(layers.Conv2D(32, 3, activation="relu", name="conv_initial", padding="same"))
            model_cnn.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pool_1", padding="valid"))
            model_cnn.add(layers.Dropout(0.2))
            model_cnn.add(layers.Flatten(name="flatten"))
            model_cnn.add(layers.Dense(128, activation="relu", name="dense_1"))
            model_cnn.add(layers.Dense(number_classes, activation="softmax", name="softmax_layer"))

            # Get the summary of network to check it is correct.
            # model_cnn.build(input_shape = x_trainn.shape)
            # model_cnn.summary()

            # Fit the CNN to training and validation data.
            callback = EarlyStopping(patience=10, restore_best_weights=True)
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
            model_cnn.compile(optimizer=opt, loss="categorical_crossentropy", metrics = [tf.keras.metrics.Recall(class_id = 0), 
                                                                             tf.keras.metrics.Recall(class_id = 1),
                                                                             tf.keras.metrics.Recall(class_id = 2)])                
            model_cnn.fit(class_weight=class_weight, x=x_trainn, y=y_train_classificationn, batch_size=32, epochs=200, callbacks=[callback], validation_data=(x_val, y_val_classification))
            
            # Evaluate performance
            y_pred_new_val_class = model_cnn.predict(x_new_val)
            y_pred_new_val = np.argmax(y_pred_new_val_class, axis=1)
            y_new_val = np.argmax(y_new_val_classification, axis=1)
            
            # print(f'y_val.shape = {y_val.shape}')
            # print(f'y_pred_new_val_class = {y_pred_val_class.shape}')
            # print(f'y_pred_new_val = {y_pred_val.shape}')
            
            conf_matrix_fold = confusion_matrix(y_new_val, y_pred_new_val)
            balanced_accuracy_fold = balanced_accuracy_score(y_new_val, y_pred_new_val)
            
            print(f"conf_matrix: {conf_matrix_fold}")
            print(f"balanced_accuracy_score: {balanced_accuracy_fold}")
            
            conf_matrix_folds.append(conf_matrix_fold)
            balanced_accuracy_folds.append(balanced_accuracy_fold)
    
    balanced_accuracy = np.average(balanced_accuracy_folds)
    conf_matrix = np.empty([number_classes, number_classes])
    for k in range(len(conf_matrix_folds)):
        for i in range(number_classes):
            for j in range(number_classes):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / (kk) * num_iterations)
    
    print(f"for k = {kk} folds")
    print(f"conf_matrix: {conf_matrix}")
    print(f"balanced_accuracy_score: {balanced_accuracy}")

def transformXimage(x, number_images, size):    
    x_image = np.empty([x.shape[0], size, size, 3])
    
    for i in range(number_images):
        counter = 0
        for j in range(size):
            for k in range(size):
                for l in range(3):
                    x_image[i][j][k][l] = x[i][counter]
                    counter += 1
    
    return x_image

def CNN(balance, mode_imbalance, number_classes):
    class_weight = {0: 1, 1: 1, 2: 1}
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')
    x_test = np.load('data/Xtest_Classification2.npy')
    print(f'x_train.shape = {x_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_train.shape = {y_train.shape}')
    
    # Convert train labels to one-hot encoding
    y_train_classification = tf.keras.utils.to_categorical(y_train, num_classes=number_classes)

    x_train, x_val, y_train_classification, y_val_classification = train_test_split(x_train, y_train_classification, test_size=0.3, random_state=1234)
    x_new_val, x_val, y_new_val_classification, y_val_classification = train_test_split(x_val, y_val_classification, test_size=0.5, random_state=1234)
    
    y_train_classification_1D = np.argmax(y_train_classification, axis=1)
    if balance: 
        if mode_imbalance == 1: # Oversampling with SMOTE with SMOTE
            sm = SMOTE(random_state=42)
            x_train, y_train_classification_1D = sm.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 5: # Oversampling with RandomOverSampler
            oversample = RandomOverSampler()
            x_train, y_train_classification_1D = oversample.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 2: # Undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train_classification_1D = undersample.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 3: # class weight
            class1_pos = np.count_nonzero(y_train_classification_1D == 1)
            class2_pos = np.count_nonzero(y_train_classification_1D == 2)
            total = y_train_classification_1D.shape[0]
            class0_pos = total - class1_pos - class2_pos
            weight_for_0 = (1 / class0_pos) * (total / 3.0)
            weight_for_1 = (1 / class1_pos) * (total / 3.0)
            weight_for_2 = (1 / class2_pos) * (total / 3.0)
            class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
            
            print('Weight for class 0: {:.2f}'.format(weight_for_0))
            print('Weight for class 1: {:.2f}'.format(weight_for_1))
            print('Weight for class 2: {:.2f}'.format(weight_for_2))
        else: # Oversampling with our implementation
            x_train, y_train_classification_1D = Oversampling(x_train, y_train_classification_1D)
            
    y_train_classification = tf.keras.utils.to_categorical(y_train_classification_1D, num_classes=number_classes)
    # print(f'x_train.shape = {x_train.shape}')
    # print(f'y_train_classification.shape = {y_train_classification.shape}')

    x_train_image = transformXimage(x_train, x_train.shape[0], 5)
    x_val_image = transformXimage(x_val, x_val.shape[0], 5)
    x_new_val_image = transformXimage(x_new_val, x_new_val.shape[0], 5)
    x_test_image = transformXimage(x_test, x_test.shape[0], 5)
    
    print(f"x_val_image.shape = {x_val_image.shape}")
    print(f"x_new_val_image.shape = {x_new_val_image.shape}")
    
    model_cnn = tf.keras.Sequential()
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical",
                            input_shape=(x_train_image.shape[1],
                                        x_train_image.shape[2],
                                        x_train_image.shape[3])),
            layers.RandomRotation(0.1),
        ]
    )
    model_cnn.add(data_augmentation)
    model_cnn.add(layers.Rescaling(1./255, input_shape=(5,5,3)))
    model_cnn.add(layers.Conv2D(32, 3, activation="relu", name="conv_initial", padding="same"))
    model_cnn.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pool_1", padding="valid"))
    model_cnn.add(layers.Dropout(0.2))
    model_cnn.add(layers.Flatten(name="flatten"))
    model_cnn.add(layers.Dense(128, activation="relu", name="dense_1"))
    model_cnn.add(layers.Dense(number_classes, activation="softmax", name="softmax_layer"))

    # Get the summary of network to check it is correct.
    # model_cnn.build(input_shape = x_trainn.shape)
    # model_cnn.summary()

    # Fit the CNN to training and validation data.
    callback = EarlyStopping(patience=10, restore_best_weights=True)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
    model_cnn.compile(optimizer=opt, loss="categorical_crossentropy", metrics = [tf.keras.metrics.Recall(class_id = 0), 
                                                                             tf.keras.metrics.Recall(class_id = 1),
                                                                             tf.keras.metrics.Recall(class_id = 2)]) 
    model_cnn.fit(class_weight=class_weight, x=x_train_image, y=y_train_classification, batch_size=32, epochs=200, callbacks=[callback], validation_data=(x_val_image, y_val_classification))
    
    # Evaluate performance
    y_pred_new_val_class = model_cnn.predict(x_new_val_image)
    y_pred_val_class = model_cnn.predict(x_val_image)
    y_pred_test_class = model_cnn.predict(x_test_image)
    
    y_pred_new_val = np.argmax(y_pred_new_val_class, axis=1)
    y_new_val = np.argmax(y_new_val_classification, axis=1)
    
    y_pred_val = np.argmax(y_pred_val_class, axis=1)
    y_val = np.argmax(y_val_classification, axis=1)

    y_pred_test = np.argmax(y_pred_test_class, axis=1)
    
    conf_matrix = confusion_matrix(y_new_val, y_pred_new_val)
    balanced_accuracy = balanced_accuracy_score(y_new_val, y_pred_new_val)
    
    conf_matrix2 = confusion_matrix(y_val, y_pred_val)
    balanced_accuracy2 = balanced_accuracy_score(y_val, y_pred_val)
    
    print(f"Irrelevant: conf_matrix of validation data: {conf_matrix2}")
    print(f"Irrelevant: balanced_accuracy_score of validation data: {balanced_accuracy2}")
    
    print(f"conf_matrix of new_validation data: {conf_matrix}")
    print(f"balanced_accuracy_score of new_validation data: {balanced_accuracy}")
    
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_pred_test.shape = {y_pred_test.shape}')
    
    np.save('data/ytest_Classification2', y_pred_test)

def VisualizeImage():
    x_test = np.load('data/Xtest_Classification2.npy') 

    x_test_image = x_test.reshape(x_test.shape[0], 5, 5, 3)

    # Images from the testing data
    imgs_test = [x_test_image[i] for i in range(30)]

    for i in range(len(imgs_test)):
        plt.subplot(6,5,i+1)
        plt.imshow(imgs_test[i])
    plt.show()

def RandomForest_kfolds(kk, balance, mode_imbalance, number_classes): 
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')
    x_train = x_train / 255 
    
    # Cross validation to search for the hyperparameters
    rf = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_samples=50)
    parameters_rf = {"criterion":["gini", "entropy"], "min_samples_leaf":[1, 2, 3], "max_depth":[None, 10, 100], 
                "min_samples_split":[2, 3, 4], "max_features":["sqrt", "log2"]}
    rfc = GridSearchCV(rf, parameters_rf, cv=10)
    rfc.fit(x_train, y_train)

    rfc_kfolds = rfc.best_params_
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    balanced_accuracy_folds = []
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x_train, y_train):
        x_val = x_train[test]
        y_val = y_train[test]
        x_trainn = x_train[train]
        y_trainn = y_train[train]
        
        if balance: 
            if mode_imbalance == 1: # Oversampling with SMOTE
                sm = SMOTE(random_state=42)
                x_trainn, y_trainn = sm.fit_resample(x_trainn, y_trainn)
            elif mode_imbalance == 5: # Oversampling with RandomOverSampler
                oversample = RandomOverSampler()
                x_trainn, y_trainn = oversample.fit_resample(x_trainn, y_trainn)
            elif mode_imbalance == 2: # Undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_trainn = undersample.fit_resample(x_trainn, y_trainn)
            else: # Oversampling with our implementation
                x_trainn, y_trainn = Oversampling(x_trainn, y_trainn)
                
        print(f"Fold no = {fold_no}")
        fold_no += 1
        
        rf_kfold = RandomForestClassifier(criterion = rfc_kfolds["criterion"], min_samples_leaf = rfc_kfolds["min_samples_leaf"], 
                                    max_depth = rfc_kfolds["max_depth"], min_samples_split = rfc_kfolds["min_samples_split"], 
                                    max_features = rfc_kfolds["max_features"], n_estimators = 100, bootstrap = True, max_samples=50)
        rf_kfold.fit(x_trainn, y_trainn)
        
        y_pred_val = rf_kfold.predict(x_val)
        
        # print(f'y_val.shape = {y_val.shape}')
        # print(f'y_pred_val.shape = {y_pred_val.shape}')
        # print(f'y_trainn.shape = {y_trainn.shape}')
        
        conf_matrix_fold = confusion_matrix(y_val, y_pred_val)
        balanced_accuracy_fold = balanced_accuracy_score(y_val, y_pred_val)
        
        # print(f"conf_matrix: {conf_matrix_fold}")
        # print(f"balanced_accuracy_score: {balanced_accuracy_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        balanced_accuracy_folds.append(balanced_accuracy_fold)
    
    balanced_accuracy = np.average(balanced_accuracy_folds)
    conf_matrix = np.empty([number_classes, number_classes])
    for k in range(len(conf_matrix_folds)):
        for i in range(number_classes):
            for j in range(number_classes):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    
    print(f"for k = {kk} folds")
    print(f"conf_matrix: {conf_matrix}")
    print(f"balanced_accuracy_score: {balanced_accuracy}")

def RandomForest(balance, mode_imbalance, number_classes):
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')
    x_test = np.load('data/Xtest_Classification2.npy')
    x_train = x_train / 255 
    x_test = x_test / 255 
    
    if balance: 
        if mode_imbalance == 1: # Oversampling with SMOTE
            sm = SMOTE(random_state=42)
            x_train, y_train = sm.fit_resample(x_train, y_train)
        elif mode_imbalance == 5: # Oversampling with RandomOverSampler
            oversample = RandomOverSampler()
            x_train, y_train = oversample.fit_resample(x_train, y_train)
        elif mode_imbalance == 2: # Undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train = undersample.fit_resample(x_train, y_train)
        else: # Oversampling with our implementation
            x_train, y_train = Oversampling(x_train, y_train)
    
    # Cross validation to search for the hyperparameters
    rf = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_samples=50)
    parameters_rf = {"criterion":["gini", "entropy"], "min_samples_leaf":[1, 2, 3], "max_depth":[None, 10, 100], 
                "min_samples_split":[2, 3, 4], "max_features":["sqrt", "log2"]}
    rfc = GridSearchCV(rf, parameters_rf, cv=10)
    rfc.fit(x_train, y_train)

    y_pred_train = rfc.predict(x_train)
    y_pred_test = rfc.predict(x_test)
    
    print(f'y_train.shape = {y_train.shape}')
    print(f'y_pred_train.shape = {y_pred_train.shape}')
    print(f'y_pred_test.shape = {y_pred_test.shape}')
    
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    balanced_accuracy = balanced_accuracy_score(y_train, y_pred_train)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - balanced_accuracy_score of train data: {balanced_accuracy}")

    np.save('data/ytest_Classification2', y_pred_test)
    
def SVM(mode, balance, mode_imbalance, number_classes):
    x_train = np.load('data/Xtrain_Classification2.npy') 
    y_train = np.load('data/Ytrain_Classification2.npy')
    x_train = x_train / 255 
    
    if balance: 
        if mode_imbalance == 1: # Oversampling with SMOTE
            sm = SMOTE(random_state=42)
            x_train, y_train = sm.fit_resample(x_train, y_train)
        elif mode_imbalance == 5: # Oversampling with RandomOverSampler
            oversample = RandomOverSampler()
            x_train, y_train = oversample.fit_resample(x_train, y_train)
        elif mode_imbalance == 2: # Undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train = undersample.fit_resample(x_train, y_train)
        else: # Oversampling with our implementation
            x_train, y_train = Oversampling(x_train, y_train)
            
    np.random.seed(0)
    np.random.shuffle(x_train)
    np.random.seed(0)
    np.random.shuffle(y_train)
    
    # 80% train data + 20% validation data
    perc = int(x_train.shape[0] * 0.8)
    x_val = x_train[perc:]
    y_val = y_train[perc:]
    
    x_train = x_train[0:perc]
    y_train = y_train[0:perc]
    
    x_test = np.load('data/Xtest_Classification2.npy')
    x_test = x_test / 255 
    
    # Cross validation to search for the hyperparameters
    if mode == 1:
        svmm = svm.SVC(kernel='linear', decision_function_shape='ovo', class_weight='balanced', C = 1)
        # parameters = {"C" : [0.01, 1, 10]}, obtained C = 1
        
    elif mode == 2:
        svmm = svm.SVC(kernel='rbf', decision_function_shape='ovo', class_weight='balanced', C = 1, gamma = 'scale')
        # parameters = {"C":[0.01, 1, 10], "gamma":['scale', 'auto']}, obtained C = 1, gamma = 'scale'
    else:
        svmm = svm.SVC(kernel='poly', decision_function_shape='ovo', class_weight='balanced', C = 1, degree = 0.1, coef0 = 2)
        # parameters = {"C":[0.01, 1, 10], "degree":[2, 3, 4], "coef0":[0.1, 1, 10]}, obtained C = 1, degree = 0.1, coef0 = 2
        
    # svmm = GridSearchCV(svmm, parameters, cv=10, verbose = 3)
    svmm.fit(x_train, y_train)
    # print(svmm.best_params_)

    y_pred_train = svmm.predict(x_train)
    y_pred_test = svmm.predict(x_test)
    y_pred_val = svmm.predict(x_val)
    
    print(f'y_train.shape = {y_train.shape}')
    print(f'y_pred_train.shape = {y_pred_train.shape}')
    print(f'y_pred_test.shape = {y_pred_test.shape}')
    print(f'y_pred_val.shape = {y_pred_val.shape}')
    
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    balanced_accuracy = balanced_accuracy_score(y_train, y_pred_train)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - balanced_accuracy_score of train data: {balanced_accuracy}")
    
    conf_matrix2 = confusion_matrix(y_val, y_pred_val)
    balanced_accuracy2 = balanced_accuracy_score(y_val, y_pred_val)
    
    print(f"conf_matrix of validation data: {conf_matrix2}")
    print(f"balanced_accuracy_score of validation data: {balanced_accuracy2}")

    np.save('data/ytest_Classification2', y_pred_test)

def Oversampling(x, y):
    class1_pos = np.count_nonzero(y == 1)
    class2_pos = np.count_nonzero(y == 2)
    class0_pos = np.count_nonzero(y == 0)
    print("Elements for class 0:", class0_pos)
    print("Elements for class 1:", class1_pos)
    print("Elements for class 2:", class2_pos)
    print(x.shape)
    counter1 = 0
    counter2 = 0
    rangee = x.shape[0]
    racio1 = class1_pos - (int)(math.floor(class1_pos/class2_pos)) * class2_pos
    racio2 = class1_pos - (int)(math.floor(class1_pos/class0_pos)) * class0_pos

    for i in range(rangee):
        aux_x = x[i]
        aux_xt = np.transpose(aux_x)
        aux_y = y[i]
        if i % 1000 == 0:
            print('Loading oversampling: ', 100 * i / rangee, '%')
        if aux_y == 2:
            counter1 += 1
            if counter1 <= racio1:
                x = np.append(x, [aux_xt, aux_x * 0.5, aux_xt * 0.5, aux_x * 0.8], axis = 0)
                y = np.append(y, [aux_y, aux_y, aux_y, aux_y], axis = 0) 
            else:
                x = np.append(x, [aux_xt, aux_x * 0.5, aux_xt * 0.5], axis = 0)
                y = np.append(y, [aux_y, aux_y, aux_y], axis = 0)  
        if aux_y == 0:
            counter2 += 1
            if counter2 <= racio2:
                x = np.append(x, [aux_x * 0.1, aux_xt * 0.1, aux_x * 0.2, aux_xt * 0.2,
                                            aux_x * 0.3, aux_xt * 0.3, aux_x * 0.4, aux_xt * 0.4,
                                            aux_x * 0.5, aux_xt * 0.5, aux_x * 0.6, aux_xt * 0.6,
                                            aux_x * 0.7, aux_xt * 0.7, aux_x * 0.8, aux_xt * 0.8,
                                            aux_x * 0.9, aux_xt * 0.9, aux_x * 0.55, aux_xt * 0.55, 
                                            aux_x * 0.45, aux_xt * 0.45, aux_x * 0.35, aux_xt * 0.35, 
                                            aux_x * 0.85], axis = 0)
                y = np.append(y, [aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y,
                                            aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y,
                                            aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y], axis = 0) 
            else:  
                x = np.append(x, [aux_x * 0.1, aux_xt * 0.1, aux_x * 0.2, aux_xt * 0.2,
                                                aux_x * 0.3, aux_xt * 0.3, aux_x * 0.4, aux_xt * 0.4,
                                                aux_x * 0.5, aux_xt * 0.5, aux_x * 0.6, aux_xt * 0.6,
                                                aux_x * 0.7, aux_xt * 0.7, aux_x * 0.8, aux_xt * 0.8,
                                                aux_x * 0.9, aux_xt * 0.9, aux_x * 0.55, aux_xt * 0.55, 
                                                aux_x * 0.45, aux_xt * 0.45, aux_x * 0.35, aux_xt * 0.35], axis = 0)
                y = np.append(y, [aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y,
                                                aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y,
                                                aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y, aux_y], axis = 0)
            
    print("Elements for class 0:", np.count_nonzero(y == 0))
    print("Elements for class 1:", np.count_nonzero(y == 1))
    print("Elements for class 2:", np.count_nonzero(y == 2))
    print(x.shape)
    
    return [x, y]
    
# All models tested

# Gaussian
# NaiveBayes_kfolds(5, 1, True, 4, 3)
# NaiveBayesclassifier(1, False, 1, 3)

# Multinomial
# NaiveBayes_kfolds(5, 2, True, 5, 3)
# NaiveBayesclassifier(2, False, 1, 3)

# Complement
# NaiveBayes_kfolds(5, 3, True, 5, 3)
# NaiveBayesclassifier(3, False, 1, 3)

# Bernoulli
# NaiveBayes_kfolds(5, 4, True, 4, 3)
# NaiveBayesclassifier(4, True, 3, 3)

# KNeighborsClassifier_testing(11, 50, True, 1, 3)
# KNeighborsclassifier(3, True, 1)
 
# MLP_testing(6, 3, True, 32, True, 2, 3)
# MLP(8, True, 32, False, 0, 3)
# MLP_testing(15, 3, False, 32, True, 5, 3)
# MLP(17, False, 32, False, 0, 3)
# MLP_testing(15, 3, False, 64, True, 5, 3)
# MLP(15, False, 64, False, 0, 3)

# CNN_kfolds(3, True, 3, 1, 3)
# CNN(True, 1, 3)

# RandomForest(False, 0, 3)
# RandomForest_kfolds(5, True, 5, 3)

# Linear Kernel
# SVM(1, True, 5, 3)

# RBF Kernel
# SVM(2, True, 4, 3)

# Polynomial Kernel
# SVM(3, True, 4, 3)

# After a lot of testing, we choose KNeighborsClassifier with 3 neighbors.
# We obtain the best results using oversampling with SMOTE.
KNeighborsclassifier(3, True, 1)
