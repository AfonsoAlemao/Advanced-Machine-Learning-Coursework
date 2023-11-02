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


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

def MLP_kfolds(size_network, kk, layer_increasing, hidden_layer_start, balance, mode_imbalance):
    if size_network < 1:
        size_network = 1     
    class_weight = {0: 1, 1: 1}
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')

    # Convert train labels to one-hot encoding
    y_train_classification = tf.keras.utils.to_categorical(y_train, num_classes=2)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    f1_folds = []
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x_train, y_train_classification):
        x_val = x_train[test]
        y_val_classification = y_train_classification[test]
        x_trainn = x_train[train]
        y_train_classificationn = y_train_classification[train]
        
        print(f"Fold no = {fold_no}")
        fold_no += 1
        
        y_train_classificationn_1D = np.argmax(y_train_classificationn, axis=1)
        if balance: 
            if mode_imbalance == 1: # oversampling
                sm = SMOTE(random_state=42)
                x_trainn, y_train_classificationn_1D = sm.fit_resample(x_trainn, y_train_classificationn_1D)
            elif mode_imbalance == 2: # undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_train_classificationn_1D = undersample.fit_resample(x_trainn, y_train_classificationn_1D)
            else: # class weight 
                pos = np.count_nonzero(y_train_classificationn_1D == 1)
                total = y_train_classificationn_1D.shape[0]
                neg = total - pos
                weight_for_0 = (1 / neg) * (total / 2.0)
                weight_for_1 = (1 / pos) * (total / 2.0)
                class_weight = {0: weight_for_0, 1: weight_for_1}
                
                print('Weight for class 0: {:.2f}'.format(weight_for_0))
                print('Weight for class 1: {:.2f}'.format(weight_for_1))
        y_train_classificationn = tf.keras.utils.to_categorical(y_train_classificationn_1D, num_classes=2)

        # Sequential model. Start by adding a flatten layer to convert the 3D images to 1D. 
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(2700,1), name="initial"))
        
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
        model.add(layers.Dense(2, activation="softmax", name="final"))

        # Get the summary of network to check it is correct.
        model.build(input_shape = x_trainn.shape)
        # model.summary()

        # Create an Early stopping monitor that will stop training when the validation loss is not improving
        callback = EarlyStopping(patience=10, restore_best_weights=True)

        # Fit the MLP to training and validation data using ’categorical_crossentropy’ as the loss function, 
        # a batch size of 200 (mini-batch) and Adam as the optimizer (learning rate=0.001, clipnorm=1). 
        # Choose, as stopping criterion, the number of epochs reaching 200. 
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = [tf.keras.metrics.Precision(class_id = 1),tf.keras.metrics.Recall(class_id = 1)])        
        model.fit(class_weight=class_weight, verbose=0, x=x_trainn, y=y_train_classificationn, batch_size=200, epochs=200, callbacks=[callback], validation_data=(x_val, y_val_classification))
        # Evaluate performance 
        y_pred_classification = model.predict(x_val)
        
        for i in range(y_pred_classification.shape[0]):
            if (y_pred_classification[i][0] >= y_pred_classification[i][1]):
                y_pred_classification[i][0] = 1
                y_pred_classification[i][1] = 0
            else:
                y_pred_classification[i][0] = 0
                y_pred_classification[i][1] = 1
        
        y_pred = np.argmax(y_pred_classification, axis=1)
        y_val = np.argmax(y_val_classification, axis=1)
        
        # print(f'y_val.shape = {y_val.shape}')
        # print(f'y_val_classification.shape = {y_val_classification.shape}')
        # print(f'y_pred = {y_pred.shape}')
        # print(f'y_pred_classification = {y_pred_classification.shape}')
        # print(f'y_train_classification.shape = {y_train_classification.shape}')
        
        conf_matrix_fold = confusion_matrix(y_val, y_pred)
        f1_fold = f1_score(y_val, y_pred)
        
        # print(f"conf_matrix: {conf_matrix_fold}")
        # print(f"f1_score: {f1_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        f1_folds.append(f1_fold)
    
    f1 = np.average(f1_folds)
    conf_matrix = np.empty([2, 2])
    for k in range(len(conf_matrix_folds)):
        for i in range(2):
            for j in range(2):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    return [f1, conf_matrix]


def MLP_testing(max_number_layers, kk, layer_increasing, hidden_layer_start, balance, mode_imbalance):
    print(f"Using kfolds = {kk}")
    f1 = []
    confusion_matrix = []
    for i in range(max_number_layers):
        (f1_i, confusion_matrix_i) = MLP_kfolds(i + 1, kk, layer_increasing, hidden_layer_start, balance, mode_imbalance)
        if (i == 0):
            print(f"For {i + 1} layer:")
        else:
            print(f"For {i + 1} layers:")
        print(f"Average conf_matrix: {confusion_matrix_i}")
        print(f"Average f1_score: {f1_i}")
        f1.append(f1_i)
        confusion_matrix.append(confusion_matrix_i)
    best_num_layer = (f1.index(max(f1)) + 1)
    print(f'Best case: Number of layers = {best_num_layer} with f1 = {max(f1)}')
    
def MLP(size_network, layer_increasing, hidden_layer_start, balance, mode_imbalance):
    class_weight = {0: 1, 1: 1}
    if size_network < 1:
        size_network = 1     
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
    x_test = np.load('data/Xtest_Classification1.npy')

    # Convert train labels to one-hot encoding
    y_train_classification = tf.keras.utils.to_categorical(y_train, num_classes=2)
    
    # Split train data in 80% train data and 20% validation data
    x_train, x_val, y_train_classification, y_val_classification = train_test_split(x_train, y_train_classification, test_size=0.2, random_state=1234)
    
    y_train_classification_1D = np.argmax(y_train_classification, axis=1)
    if balance: 
        if mode_imbalance == 1: # oversampling
            sm = SMOTE(random_state=42)
            x_train, y_train_classification_1D = sm.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 2: # undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train_classification_1D = undersample.fit_resample(x_train, y_train_classification_1D)
        else: # class weight 
            pos = np.count_nonzero(y_train_classification_1D == 1)
            total = y_train_classification_1D.shape[0]
            neg = total - pos
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}
            
            print('Weight for class 0: {:.2f}'.format(weight_for_0))
            print('Weight for class 1: {:.2f}'.format(weight_for_1))
    y_train_classification = tf.keras.utils.to_categorical(y_train_classification_1D, num_classes=2)
    
    # Sequential model. Start by adding a flatten layer to convert the 3D images to 1D. 
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(2700,1), name="initial"))
    
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
    model.add(layers.Dense(2, activation="softmax", name="final"))

    # Get the summary of network to check it is correct.
    model.build(input_shape = x_train.shape)
    model.summary()

    # Create an Early stopping monitor that will stop training when the validation loss is not improving
    callback = EarlyStopping(patience=10, restore_best_weights=True)

    # Fit the MLP to training and validation data using ’categorical_crossentropy’ as the loss function, 
    # a batch size of 200 (mini-batch) and Adam as the optimizer (learning rate=0.001, clipnorm=1). 
    # Choose, as stopping criterion, the number of epochs reaching 200. 
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = [tf.keras.metrics.Precision(class_id = 1),tf.keras.metrics.Recall(class_id = 1)])        

    model.fit(class_weight=class_weight, x=x_train, y=y_train_classification, batch_size=200, epochs=200, callbacks=[callback], validation_data=(x_val, y_val_classification))
    
    # Evaluate performance 
    y_val_pred_classification = model.predict(x_val)
    y_test_pred_classification = model.predict(x_test)
    for i in range(y_val_pred_classification.shape[0]):
        if (y_val_pred_classification[i][0] >= y_val_pred_classification[i][1]):
            y_val_pred_classification[i][0] = 1
            y_val_pred_classification[i][1] = 0
        else:
            y_val_pred_classification[i][0] = 0
            y_val_pred_classification[i][1] = 1
    
    for i in range(y_test_pred_classification.shape[0]):
        if (y_test_pred_classification[i][0] >= y_test_pred_classification[i][1]):
            y_test_pred_classification[i][0] = 1
            y_test_pred_classification[i][1] = 0
        else:
            y_test_pred_classification[i][0] = 0
            y_test_pred_classification[i][1] = 1
    
    y_pred_val = np.argmax(y_val_pred_classification, axis=1)
    y_val = np.argmax(y_val_classification, axis=1)
    y_pred_test = np.argmax(y_test_pred_classification, axis=1)
    
    print(f'y_pred_train = {y_pred_val.shape}')
    print(f'y_train_pred_classification = {y_val_pred_classification.shape}')
    print(f'y_pred_test = {y_pred_test.shape}')
    print(f'y_test_pred_classification = {y_test_pred_classification.shape}')
    
    conf_matrix = confusion_matrix(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - f1_score of train data: {f1}")
    
    np.save('data/ytest_Classification1', y_pred_test)
    

def KNeighborsClassifier_kfolds(neighbors, kk, balance, mode_imbalance):
    if neighbors < 1:
        neighbors = 5    
    x_train = np.load('data/Xtrain_Classification1.npy')
    x_train = x_train / 255 
    y_train = np.load('data/ytrain_Classification1.npy')

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    f1_folds = []
    
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
            if mode_imbalance == 1: # oversampling
                sm = SMOTE(random_state=42)
                x_trainn, y_trainn = sm.fit_resample(x_trainn, y_trainn)
            else: # undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_trainn = undersample.fit_resample(x_trainn, y_trainn)

        
        neigh = KNeighborsClassifier(n_neighbors=neighbors)
        neigh.fit(x_trainn, y_trainn)
        y_pred_val = neigh.predict(x_val)
        
        # print(f'y_val.shape = {y_val.shape}')
        # print(f'y_pred_val.shape = {y_pred_val.shape}')
        # print(f'y_trainn.shape = {y_trainn.shape}')
        
        conf_matrix_fold = confusion_matrix(y_val, y_pred_val)
        f1_fold = f1_score(y_val, y_pred_val)
        
        # print(f"conf_matrix: {conf_matrix_fold}")
        # print(f"f1_score: {f1_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        f1_folds.append(f1_fold)
    
    f1 = np.average(f1_folds)
    conf_matrix = np.empty([2, 2])
    for k in range(len(conf_matrix_folds)):
        for i in range(2):
            for j in range(2):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    return [f1, conf_matrix]


def KNeighborsClassifier_testing(max_number_neighbors, kk, balance, mode_imbalance):
    print(f"Using kfolds = {kk}")
    f1 = []
    confusion_matrix = []
    for i in range(max_number_neighbors):
        (f1_i, confusion_matrix_i) = KNeighborsClassifier_kfolds(i + 1, kk, balance, mode_imbalance)
        if (i == 0):
            print(f"For {i + 1} neighbor:")
        else:
            print(f"For {i + 1} neighbors:")
        # print(f"Average conf_matrix: {confusion_matrix_i}")
        print(f"Average f1_score: {f1_i}")
        f1.append(f1_i)
        confusion_matrix.append(confusion_matrix_i)
    best_num_neighbors = (f1.index(max(f1)) + 1)
    print(f'Best case: Number of neighbors = {best_num_neighbors} with f1 = {max(f1)}')

def KNeighborsclassifier(neighbors, balance, mode_imbalance):
    if neighbors < 1:
        neighbors = 5    
    x_train = np.load('data/Xtrain_Classification1.npy') 
    x_train = x_train / 255 
    y_train = np.load('data/ytrain_Classification1.npy')
    x_test = np.load('data/Xtest_Classification1.npy')
    x_test = x_test / 255 

    if balance: 
        if mode_imbalance == 1: # oversampling
            sm = SMOTE(random_state=42)
            x_train, y_train = sm.fit_resample(x_train, y_train)
        else: # undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train = undersample.fit_resample(x_train, y_train)
                
    neigh = KNeighborsClassifier(n_neighbors=neighbors)
    neigh.fit(x_train, y_train)
    y_pred_train = neigh.predict(x_train)
    y_pred_test = neigh.predict(x_test)
    
    print(f'y_train.shape = {y_train.shape}')
    print(f'y_pred_train.shape = {y_pred_train.shape}')
    print(f'y_pred_test.shape = {y_pred_test.shape}')
    print(f'x_test.shape = {x_test.shape}')
    
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - f1_score of train data: {f1}")
    
    np.save('data/ytest_Classification1', y_pred_test)

def NaiveBayes_kfolds(kk, mode, balance, mode_imbalance):
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
    x_train = x_train / 255 
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    f1_folds = []
    
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
            if mode_imbalance == 1: # oversampling
                sm = SMOTE(random_state=42)
                x_trainn, y_trainn = sm.fit_resample(x_trainn, y_trainn)
            else: # undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_trainn = undersample.fit_resample(x_trainn, y_trainn)
                
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
        f1_fold = f1_score(y_val, y_pred_val)
        
        # print(f"conf_matrix: {conf_matrix_fold}")
        # print(f"f1_score: {f1_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        f1_folds.append(f1_fold)
    
    f1 = np.average(f1_folds)
    conf_matrix = np.empty([2, 2])
    for k in range(len(conf_matrix_folds)):
        for i in range(2):
            for j in range(2):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    
    print(f"for k = {kk} folds")
    print(f"conf_matrix: {conf_matrix}")
    print(f"f1_score: {f1}")

def NaiveBayesclassifier(mode, balance, mode_imbalance):
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
    x_test = np.load('data/Xtest_Classification1.npy')
    x_train = x_train / 255 
    x_test = x_test / 255 
    
    if balance: 
        if mode_imbalance == 1: # oversampling
            sm = SMOTE(random_state=42)
            x_train, y_train = sm.fit_resample(x_train, y_train)
        else: # undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train = undersample.fit_resample(x_train, y_train)
                
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
    f1 = f1_score(y_train, y_pred_train)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - f1_score of train data: {f1}")

    np.save('data/ytest_Classification1', y_pred_test)

def CNN_kfolds(kk, balance, mode_imbalance, num_iterations):  
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
    class_weight = {0: 1, 1: 1}
    
    # Convert train labels to one-hot encoding
    y_train_classification = tf.keras.utils.to_categorical(y_train, num_classes=2)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    f1_folds = []
    
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
                if mode_imbalance == 1: # oversampling with SMOTE
                    sm = SMOTE(random_state=42)
                    x_trainn, y_train_classificationn_1D = sm.fit_resample(x_trainn, y_train_classificationn_1D)
                elif mode_imbalance == 2: # undersampling 
                    undersample = RandomUnderSampler(sampling_strategy='majority')
                    x_trainn, y_train_classificationn_1D = undersample.fit_resample(x_trainn, y_train_classificationn_1D)
                elif mode_imbalance == 3: # class weight
                    pos = np.count_nonzero(y_train_classificationn_1D == 1)
                    total = y_train_classificationn_1D.shape[0]
                    neg = total - pos
                    weight_for_0 = (1 / neg) * (total / 2.0)
                    weight_for_1 = (1 / pos) * (total / 2.0)
                    class_weight = {0: weight_for_0, 1: weight_for_1}
                    
                    print('Weight for class 0: {:.2f}'.format(weight_for_0))
                    print('Weight for class 1: {:.2f}'.format(weight_for_1))
                else: # oversampling without SMOTE
                    counter = 0
                    for i in range(x_trainn.shape[0]):
                        if y_train_classificationn_1D[i] == 1:
                            counter += 1
                            np.append(x_trainn, np.transpose(x_trainn[i]))
                            np.append(y_train_classificationn_1D, y_train_classificationn_1D[i])  
                            if (counter <= 779):
                                np.append(x_trainn, x_trainn[i] * 0.5)
                                np.append(y_train_classificationn_1D, y_train_classificationn_1D[i])  

            y_train_classificationn = tf.keras.utils.to_categorical(y_train_classificationn_1D, num_classes=2)

            x_trainn = transformXimage(x_trainn, x_trainn.shape[0])
            x_val = transformXimage(x_val, x_val.shape[0])
            x_new_val = transformXimage(x_new_val, x_new_val.shape[0])

            model_cnn = tf.keras.Sequential()
            data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip("horizontal_and_vertical",
                                    input_shape=(x_trainn.shape[1],
                                                x_trainn.shape[2],
                                                x_trainn.shape[3])),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1),
                    layers.RandomBrightness(0.2),
                    layers.RandomContrast(0.2),
                ]
            )
            model_cnn.add(data_augmentation)
            model_cnn.add(layers.Rescaling(1./255))
            model_cnn.add(layers.Conv2D(16, (3,3), input_shape=(30,30,3), activation="relu", name="conv_initial"))
            model_cnn.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pool_1"))
            model_cnn.add(layers.Conv2D(32, (3,3), input_shape=(30,30,3), activation="relu", name="conv_1"))
            model_cnn.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pool_2"))
            model_cnn.add(layers.Conv2D(64, (3,3), activation="relu", name="conv_2"))
            model_cnn.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pool_3"))
            model_cnn.add(layers.Dropout(0.2))
            model_cnn.add(layers.Flatten(name="flatten"))
            model_cnn.add(layers.Dense(128, activation="relu", name="dense_1"))
            model_cnn.add(layers.Dense(2, activation="softmax", name="softmax_layer"))

            # Get the summary of network to check it is correct.
            # model_cnn.build(input_shape = x_trainn.shape)
            # model_cnn.summary()

            # Fit the CNN to training and validation data.
            callback = EarlyStopping(patience=10, restore_best_weights=True)
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
            model_cnn.compile(optimizer=opt, loss="categorical_crossentropy", metrics = [tf.keras.metrics.Precision(class_id = 1), tf.keras.metrics.Recall(class_id = 1)])
            
            model_cnn.fit(class_weight=class_weight, x=x_trainn, y=y_train_classificationn, batch_size=32, epochs=200, callbacks=[callback], validation_data=(x_val, y_val_classification))
            
            # Evaluate performance
            y_pred_new_val_class = model_cnn.predict(x_new_val)
            y_pred_new_val = np.argmax(y_pred_new_val_class, axis=1)
            y_new_val = np.argmax(y_new_val_classification, axis=1)
            
            # print(f'y_val.shape = {y_val.shape}')
            # print(f'y_pred_new_val_class = {y_pred_val_class.shape}')
            # print(f'y_pred_new_val = {y_pred_val.shape}')
            
            conf_matrix_fold = confusion_matrix(y_new_val, y_pred_new_val)
            f1_fold = f1_score(y_new_val, y_pred_new_val)
            
            print(f"conf_matrix: {conf_matrix_fold}")
            print(f"f1_score: {f1_fold}")
            
            conf_matrix_folds.append(conf_matrix_fold)
            f1_folds.append(f1_fold)
    
    f1 = np.average(f1_folds)
    conf_matrix = np.empty([2, 2])
    for k in range(len(conf_matrix_folds)):
        for i in range(2):
            for j in range(2):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / (kk) * num_iterations)
    
    print(f"for k = {kk} folds")
    print(f"conf_matrix: {conf_matrix}")
    print(f"f1_score: {f1}")

def CNN(balance, mode_imbalance):
    class_weight = {0: 1, 1: 1}
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
    x_test = np.load('data/Xtest_Classification1.npy')
    print(f'y_train.shape = {y_train.shape}')
    
    # Convert train labels to one-hot encoding
    y_train_classification = tf.keras.utils.to_categorical(y_train, num_classes=2)

    x_train, x_val, y_train_classification, y_val_classification = train_test_split(x_train, y_train_classification, test_size=0.3, random_state=1234)
    x_new_val, x_val, y_new_val_classification, y_val_classification = train_test_split(x_val, y_val_classification, test_size=0.5, random_state=1234)
    
    y_train_classification_1D = np.argmax(y_train_classification, axis=1)
    if balance: 
        if mode_imbalance == 1: # oversampling with SMOTE
            sm = SMOTE(random_state=42)
            x_train, y_train_classification_1D = sm.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 2: # undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train_classification_1D = undersample.fit_resample(x_train, y_train_classification_1D)
        elif mode_imbalance == 3: # class weight
            pos = np.count_nonzero(y_train_classification_1D == 1)
            total = y_train_classification_1D.shape[0]
            neg = total - pos
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}
            
            print('Weight for class 0: {:.2f}'.format(weight_for_0))
            print('Weight for class 1: {:.2f}'.format(weight_for_1))
        else: # oversampling without SMOTE
            counter = 0
            for i in range(x_train.shape[0]):
                if y_train_classification_1D[i] == 1:
                    counter += 1
                    np.append(x_train, np.transpose(x_train[i]))
                    np.append(y_train_classification_1D, y_train_classification_1D[i])  
                    if (counter <= 779):
                        np.append(x_train, x_train[i] * 0.5)
                        np.append(y_train_classification_1D, y_train_classification_1D[i])  
            
    y_train_classification = tf.keras.utils.to_categorical(y_train_classification_1D, num_classes=2)
    # print(f'x_train.shape = {x_train.shape}')
    # print(f'y_train_classification.shape = {y_train_classification.shape}')

    x_train_image = transformXimage(x_train, x_train.shape[0])
    x_val_image = transformXimage(x_val, x_val.shape[0])
    x_new_val_image = transformXimage(x_new_val, x_new_val.shape[0])
    x_test_image = transformXimage(x_test, x_test.shape[0])

    model_cnn = tf.keras.Sequential()
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical",
                            input_shape=(x_train_image.shape[1],
                                        x_train_image.shape[2],
                                        x_train_image.shape[3])),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ]
    )
    model_cnn.add(data_augmentation)
    model_cnn.add(layers.Rescaling(1./255))
    model_cnn.add(layers.Conv2D(16, (3,3), input_shape=(30,30,3), activation="relu", name="conv_initial"))
    model_cnn.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pool_1"))
    model_cnn.add(layers.Conv2D(32, (3,3), input_shape=(30,30,3), activation="relu", name="conv_1"))
    model_cnn.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pool_2"))
    model_cnn.add(layers.Conv2D(64, (3,3), activation="relu", name="conv_2"))
    model_cnn.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pool_3"))
    model_cnn.add(layers.Dropout(0.2))
    model_cnn.add(layers.Flatten(name="flatten"))
    model_cnn.add(layers.Dense(128, activation="relu", name="dense_1"))
    model_cnn.add(layers.Dense(2, activation="softmax", name="softmax_layer"))

    # Get the summary of network to check it is correct.
    # model_cnn.build(input_shape = x_trainn.shape)
    # model_cnn.summary()

    # Fit the CNN to training and validation data.
    callback = EarlyStopping(patience=10, restore_best_weights=True)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
    model_cnn.compile(optimizer=opt, loss="categorical_crossentropy", metrics = [tf.keras.metrics.Precision(class_id = 1),tf.keras.metrics.Recall(class_id = 1)])        

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
    f1 = f1_score(y_new_val, y_pred_new_val)
    
    conf_matrix2 = confusion_matrix(y_val, y_pred_val)
    f12 = f1_score(y_val, y_pred_val)
    
    print(f"Irrelevant: conf_matrix of validation data: {conf_matrix2}")
    print(f"Irrelevant: f1_score of validation data: {f12}")
    
    print(f"conf_matrix of new_validation data: {conf_matrix}")
    print(f"f1_score of new_validation data: {f1}")
    
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_pred_test.shape = {y_pred_test.shape}')
    
    np.save('data/ytest_Classification1', y_pred_test)

def VisualizeImage():
    x_test = np.load('data/Xtest_Classification1.npy') 

    x_test_image = x_test.reshape(x_test.shape[0], 30, 30, 3)

    # Images from the testing data
    imgs_test = [x_test_image[i] for i in range(30)]

    for i in range(len(imgs_test)):
        plt.subplot(6,5,i+1)
        im = plt.imshow(imgs_test[i])
    plt.show()
    
def transformXimage(x, number_images):    
    x_image = np.empty([x.shape[0], 30, 30, 3])
    
    for i in range(number_images):
        counter = 0
        for j in range(30):
            for k in range(30):
                for l in range(3):
                    x_image[i][j][k][l] = x[i][counter]
                    counter += 1
    
    return x_image

def RandomForest_kfolds(kk, balance, mode_imbalance): 
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
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
    f1_folds = []
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x_train, y_train):
        x_val = x_train[test]
        y_val = y_train[test]
        x_trainn = x_train[train]
        y_trainn = y_train[train]
        
        if balance: 
            if mode_imbalance == 1: # oversampling
                sm = SMOTE(random_state=42)
                x_trainn, y_trainn = sm.fit_resample(x_trainn, y_trainn)
            else: # undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_trainn = undersample.fit_resample(x_trainn, y_trainn)
                
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
        f1_fold = f1_score(y_val, y_pred_val)
        
        # print(f"conf_matrix: {conf_matrix_fold}")
        # print(f"f1_score: {f1_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        f1_folds.append(f1_fold)
    
    f1 = np.average(f1_folds)
    conf_matrix = np.empty([2, 2])
    for k in range(len(conf_matrix_folds)):
        for i in range(2):
            for j in range(2):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    
    print(f"for k = {kk} folds")
    print(f"conf_matrix: {conf_matrix}")
    print(f"f1_score: {f1}")

def RandomForest(balance, mode_imbalance):
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
    x_test = np.load('data/Xtest_Classification1.npy')
    x_train = x_train / 255 
    x_test = x_test / 255 
    
    if balance: 
        if mode_imbalance == 1: # oversampling
            sm = SMOTE(random_state=42)
            x_train, y_train = sm.fit_resample(x_train, y_train)
        else: # undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train = undersample.fit_resample(x_train, y_train)
    
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
    f1 = f1_score(y_train, y_pred_train)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - f1_score of train data: {f1}")

    np.save('data/ytest_Classification1', y_pred_test)
    

def SVM_kfolds(kk, mode, balance, mode_imbalance): 
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
    x_train = x_train / 255 
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kk, shuffle=True)

    conf_matrix_folds = []
    f1_folds = []
    
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x_train, y_train):
        # Use the bigger set for training for speed purposes
        x_val = x_train[train]
        y_val = y_train[train]
        x_trainn = x_train[test]
        y_trainn = y_train[test]
        
        print(f"Fold no = {fold_no}")
        fold_no += 1
        
        if balance: 
            if mode_imbalance == 1: # oversampling
                sm = SMOTE(random_state=42)
                x_trainn, y_trainn = sm.fit_resample(x_trainn, y_trainn)
            else: # undersampling 
                undersample = RandomUnderSampler(sampling_strategy='majority')
                x_trainn, y_trainn = undersample.fit_resample(x_trainn, y_trainn)
            
        # Cross validation to search for the hyperparameters
        if mode == 1:
            svm_k = svm.SVC(kernel='linear', decision_function_shape='ovo', class_weight='balanced', max_iter = 700)
            parameters = {"C" : [0.01, 1, 10]}
        elif mode == 2:
            svm_k = svm.SVC(kernel='rbf', decision_function_shape='ovo', class_weight='balanced', max_iter = 700)
            parameters = {"C":[0.01, 1, 10], "gamma":['scale', 'auto']}
        else:
            svm_k = svm.SVC(kernel='poly', decision_function_shape='ovo', class_weight='balanced', max_iter = 700)
            parameters = {"C":[0.01, 1, 10], "degree":[2, 3, 4], "coef0":[0.1, 1, 10]}
            
        svm_kfold = GridSearchCV(svm_k, parameters, cv=10)
        
        svm_kfold.fit(x_trainn, y_trainn)
    
        y_pred_val = svm_kfold.predict(x_val[0:1000])
        
        # print(f'y_val.shape = {y_val.shape}')
        # print(f'y_pred_val.shape = {y_pred_val.shape}')
        # print(f'y_trainn.shape = {y_trainn.shape}')
        
        conf_matrix_fold = confusion_matrix(y_val[0:1000], y_pred_val)
        f1_fold = f1_score(y_val[0:1000], y_pred_val)
        
        # print(f"conf_matrix: {conf_matrix_fold}")
        # print(f"f1_score: {f1_fold}")
        
        conf_matrix_folds.append(conf_matrix_fold)
        f1_folds.append(f1_fold)
    
    f1 = np.average(f1_folds)
    conf_matrix = np.empty([2, 2])
    for k in range(len(conf_matrix_folds)):
        for i in range(2):
            for j in range(2):
                conf_matrix[i][j] += conf_matrix_folds[k][i][j]

    conf_matrix = conf_matrix *  (1 / kk)
    
    print(f"for k = {kk} folds")
    print(f"conf_matrix: {conf_matrix}")
    print(f"f1_score: {f1}")
    
def SVM(mode, balance, mode_imbalance):
    x_train = np.load('data/Xtrain_Classification1.npy') 
    y_train = np.load('data/ytrain_Classification1.npy')
    x_train = x_train / 255 
    
    if balance: 
        if mode_imbalance == 1: # oversampling
            sm = SMOTE(random_state=42)
            x_train, y_train = sm.fit_resample(x_train, y_train)
        else: # undersampling 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            x_train, y_train = undersample.fit_resample(x_train, y_train)
            
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
    
    x_test = np.load('data/Xtest_Classification1.npy')
    x_test = x_test / 255 
    
    # Cross validation to search for the hyperparameters
    if mode == 1:
        svmm = svm.SVC(kernel='linear', decision_function_shape='ovo', class_weight='balanced', C = 0.01)
        # parameters = {"C" : [0.01, 1, 10]}, obtained C = 0.01
    elif mode == 2:
        svmm = svm.SVC(kernel='rbf', decision_function_shape='ovo', class_weight='balanced', C = 10, gamma = 'scale', verbose = True)
        # parameters = {"C":[0.01, 1, 10], "gamma":['scale', 'auto']}, obtained C = 10, gamma = 'scale', verbose = True
    else:
        svmm = svm.SVC(kernel='poly', decision_function_shape='ovo', class_weight='balanced', C = 0.01, degree = 3, coef0 = 10)
        # parameters = {"C":[0.01, 1, 10], "degree":[2, 3, 4], "coef0":[0.1, 1, 10]}, obtained C = 0.01, degree = 3, coef0 = 10
        
    # svmm = GridSearchCV(svmm, parameters, cv=10)
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
    f1 = f1_score(y_train, y_pred_train)
    
    print(f"Irrelevant - conf_matrix of train data: {conf_matrix}")
    print(f"Irrelevant - f1_score of train data: {f1}")
    
    conf_matrix2 = confusion_matrix(y_val, y_pred_val)
    f12 = f1_score(y_val, y_pred_val)
    
    print(f"conf_matrix of validation data: {conf_matrix2}")
    print(f"f1_score of validation data: {f12}")

    np.save('data/ytest_Classification1', y_pred_test)

# All models tested

# Gaussian
# NaiveBayes_kfolds(5, 1, True, 2)
# NaiveBayesclassifier(1, False, 1)

# Multinomial
# NaiveBayes_kfolds(5, 2, True, 2)
# NaiveBayesclassifier(2, False, 1)

# Complement
# NaiveBayes_kfolds(5, 3, True, 2)
# NaiveBayesclassifier(3, False, 1)

# Bernoulli
# NaiveBayes_kfolds(5, 4, False, 1)
# NaiveBayesclassifier(4, True, 3)

# KNeighborsClassifier_testing(50, 5, False, 1)
# KNeighborsclassifier(1, False, 1)
 
# MLP_testing(8, 5, True, 32, True, 4)
# MLP(8, True, 32, True, 4)
# MLP_testing(30, 5, False, 32, True, 3)
# MLP(17, False, 32, False, 0)
# MLP_testing(15, 5, False, 64, True, 1)
# MLP(15, False, 64, False, 0)

# CNN_kfolds(3, True, 4, 2)
# CNN(True, 4)

# VisualizeImage()

# RandomForest(False, True, 3)
# RandomForest_kfolds(5, True, 2)

# Linear Kernel
# SVM(1, False, 0)
# SVM_kfolds(5, 1, True, 1)

# RBF Kernel
# SVM(2, False, 0)
# SVM_kfolds(5, 2, True, 2)

# Polynomial Kernel
# SVM(3, False, 0)
# SVM_kfolds(10, 3, False, 0)

# After a lot of testing, we choose CNN with the following parameters_
# We balanced the data using oversampling without SMOTE
CNN(True, 4)