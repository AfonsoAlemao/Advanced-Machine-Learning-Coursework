import sys
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
import numpy as np

# Linear predictor: computes y = [1 x^T] . beta
def modelAddOnesToX(x, beta):
    if beta.size == 0:
        print('Invalid model')
        sys.exit()
    
    # create X
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (j == 0):
                x_line = np.array([x[i, j]])
            else:
                v2 = np.array([x[i, j]])
                x_line = np.hstack((x_line, v2))
        
        if (i == 0):
            X = np.array([1])
            X = np.hstack((X, x_line))
        else:
            v1 = np.array([1])
            v1 = np.hstack((v1, x_line))
            X = np.vstack([X, v1])

    y = np.matmul(X, beta)
    return y

# Calculate SSE
def calculateSSE(y_real, y_pred):
    if y_real.shape != y_pred.shape:
        print(f'y_real.shape = {y_real.shape}, y_pred.shape = {y_pred.shape}')
        print('Confirm that both of your inputs have the same shape')
    else:
        sum = 0
        for i in range(y_pred.shape[0]):
            error = y_pred[i] - y_real[i]
            sum += pow(float(error), 2)
        return sum

# Calculate SSE for Lasso Regression
def calculateSSELasso(y_real, y_pred):
    if y_real.shape[0] != y_pred.shape[0]:
        print(f'y_real.shape = {y_real.shape}, y_pred.shape = {y_pred.shape}')
        print('Confirm that both of your inputs have the same shape')
    else:
        sum = 0
        for i in range(y_pred.shape[0]):
            error = y_pred[i] - y_real[i]
            sum += pow(float(error), 2)
        return sum

# Displays linear predictor equation for Simple Linear Regression
def printModelEquationLR(beta):
    string_print = 'y_pred = '
    for i in range(beta.shape[0]):
        if i == 0:
            string_print += 'beta_' + str(i)
        else:
            string_print += ' + ' + 'beta_' + str(i) + ' x_' + str(i)
    print(string_print)

# Linear predictor: computes y = x . beta + beta0
def model(x, beta, beta0):
    if beta.size == 0:
        print('Invalid model')
        sys.exit()
    
    y = np.matmul(x, beta)
    for i in range(y.shape[0]):
        y[i] += beta0[0]
        
    return y

# Predict y_train by Polynomial Model
def Poly(degree1):   
    x_train = np.load("data/Xtrain_Regression1.npy")
    y_train = np.load("data/Ytrain_Regression1.npy")
    x_test = np.load("data/Xtest_Regression1.npy")

    if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
        sys.exit()
        
    poly = PolynomialFeatures(degree=degree1)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)

    reg = linear_model.LinearRegression()

    reg.fit(x_train_poly, y_train)
    beta = np.array(reg.coef_)
    beta = np.transpose(beta)
    beta0 = np.array(reg.intercept_)
    beta = np.vstack([beta0, beta])

    # print(f'beta = {beta} shape beta = {beta.shape}')
    # print(f'shape x_train_poly = {x_train_poly.shape}')

    y_train_pred = modelAddOnesToX(x_train_poly, beta)
    SSE_train = calculateSSE(y_train, y_train_pred)
    print(f'Irrelevant: SSE_train = {SSE_train}')

    y_test_pred = modelAddOnesToX(x_test_poly, beta)
    np.save('data/Ytest_pred_Regression1.npy', y_test_pred)

# Spliting the training data into random train and test subsets to perform a model performance evaluation on Polynomial Model
def PolyTesting(degree1, test_size1):
    x_train_original = np.load("data/Xtrain_Regression1.npy")
    y_train_original = np.load("data/Ytrain_Regression1.npy")
    x_train, x_test, y_train, y_test = train_test_split(x_train_original, y_train_original, test_size = test_size1, random_state=42)

    if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
        sys.exit()
    
    poly = PolynomialFeatures(degree=degree1)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)

    reg = linear_model.LinearRegression()

    reg.fit(x_train_poly, y_train)
    beta = np.array(reg.coef_)
    beta = np.transpose(beta)
    beta0 = np.array(reg.intercept_)
    beta = np.vstack([beta0, beta])

    y_test_pred = modelAddOnesToX(x_test_poly, beta)
    SSE_test = calculateSSE(y_test, y_test_pred)
    print(f'SSE_validation_data = {SSE_test}')
    
# Spliting the training data into random train and test subsets to perform a model performance evaluation on Polynomial Regression
def Polykfolds(degree1, kk):
    x_train_original = np.load("data/Xtrain_Regression1.npy")
    y_train_original = np.load("data/Ytrain_Regression1.npy")
    poly = PolynomialFeatures(degree=degree1)
    x_train_poly = poly.fit_transform(x_train_original)
    
    SSE = []
    for i in range (100):
        # define the test condition
        cv = KFold(n_splits=kk, shuffle=True, random_state=i)
        # evaluate k value
        scores = -cross_val_score(linear_model.LinearRegression(), x_train_poly, y_train_original, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        SSE.append(scores)
    
    SSE_test = (x_train_poly.shape[0] / kk) * mean(scores)
    print(f'SSE_validation_data = {SSE_test}')

# Predict y_train by Simple Linear Regression using normal equations explicitly
def LinRegressionMath():
    x_train = np.load("data/Xtrain_Regression1.npy")
    y_train = np.load("data/Ytrain_Regression1.npy")
    x_test = np.load("data/Xtest_Regression1.npy")

    if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
        sys.exit()
        
    # create X
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            if (j == 0):
                x_line = np.array([x_train[i, j]])
            else:
                v2 = np.array([x_train[i, j]])
                x_line = np.hstack((x_line, v2))
        
        if (i == 0):
            X = np.array([1])
            X = np.hstack((X, x_line))
        else:
            v1 = np.array([1])
            v1 = np.hstack((v1, x_line))
            X = np.vstack([X, v1])

    # create y
    for i in range(y_train.shape[0]):
        if (i == 0):
            y = np.array([y_train[i]])
        else:
            v1 = np.array([y_train[i]])
            y = np.vstack([y,v1])
            
    try:
        XT = np.transpose(X)
        XT_X = np.matmul(XT, X)
        aux = np.linalg.det(XT_X)
    except:
        print('Invalid data')
        sys.exit()

    if aux == 0:
        print('(X mult XT) not invertible \nImpossible to get a model\nPossible reasons:\n-Small amount of data\n-Redundant features (linearly dependent)')
        sys.exit()
        
    else :
        try:
            beta = np.matmul(np.linalg.inv (XT_X), XT)
            beta = np.matmul(beta, y)
        except:
            print('Invalid data')
            sys.exit()

    # printModelEquationLR(beta)

    # print(f'beta = {beta}')

    y_train_pred = modelAddOnesToX(x_train, beta)
    SSE_train = calculateSSE(y_train, y_train_pred)
    print(f'Irrelevant: SSE_train = {SSE_train}')

    y_test_pred = modelAddOnesToX(x_test, beta)
    np.save('data/Ytest_pred_Regression1.npy', y_test_pred)
    
    if (y_test_pred.shape != (1000, 1)):
        print('confirm that both of your outputs have the same shape')

# Predict y_train by Simple Linear Regression
def LinRegression():
    x_train = np.load("data/Xtrain_Regression1.npy")
    y_train = np.load("data/Ytrain_Regression1.npy")
    x_test = np.load("data/Xtest_Regression1.npy")

    if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
        sys.exit()
        
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    beta = np.array(reg.coef_)
    beta = np.transpose(beta)
    beta0 = np.array(reg.intercept_)

    beta = np.vstack([beta0, beta])

    # printModelEquationLR(beta)

    # print(f'beta = {beta}')

    y_train_pred = modelAddOnesToX(x_train, beta)
    SSE_train = calculateSSE(y_train, y_train_pred)
    print(f'Irrelevant: SSE_train = {SSE_train}')

    y_test_pred = modelAddOnesToX(x_test, beta)
    np.save('data/Ytest_pred_Regression1.npy', y_test_pred)
    
    # print(f'x_train.shape = {x_train.shape}')
    # print(f'y_train.shape = {y_train.shape}')
    # print(f'x_test.shape = {x_test.shape}')
    # print(f'y_test_pred.shape = {y_test_pred.shape}')
    
    if (y_test_pred.shape != (1000, 1)):
        print('confirm that both of your outputs have the same shape')

# Spliting the training data into random train and test subsets to perform a model performance evaluation on Simple Linear Regression
def LinRegressionTesting(test_size1):
    x_train_original = np.load("data/Xtrain_Regression1.npy")
    y_train_original = np.load("data/Ytrain_Regression1.npy")

    x_train, x_test, y_train, y_test = train_test_split(x_train_original, y_train_original, test_size = test_size1, random_state=42)

    if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
        sys.exit()

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    beta = np.array(reg.coef_)
    beta = np.transpose(beta)
    beta0 = np.array(reg.intercept_)
    beta = np.vstack([beta0, beta])

    y_test_pred = modelAddOnesToX(x_test, beta)

    SSE_test = calculateSSE(y_test, y_test_pred)
    print(f'SSE_validation_data = {SSE_test}')

# Spliting the training data into random train and test subsets to perform a model performance evaluation on Simple Linear Regression
def LinRegressionTestingkfolds(kk):
    x_train_original = np.load("data/Xtrain_Regression1.npy")
    y_train_original = np.load("data/Ytrain_Regression1.npy")

    SSE = []
    for i in range (100):
        # define the test condition
        cv = KFold(n_splits=kk, shuffle=True, random_state=i)
        # evaluate k value
        scores = -cross_val_score(linear_model.LinearRegression(), x_train_original, y_train_original, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        SSE.append(scores)
    
    SSE_test = (x_train_original.shape[0] / kk) * mean(scores)
    print(f'SSE_validation_data = {SSE_test}')
    
# Predict y_train by Ridge Regression
def Ridge(alpha1):
    x_train = np.load("data/Xtrain_Regression1.npy")
    y_train = np.load("data/Ytrain_Regression1.npy")
    x_test = np.load("data/Xtest_Regression1.npy")

    if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
        sys.exit()

    reg = linear_model.Ridge(alpha=alpha1)
    reg.fit(x_train, y_train)
    beta = np.array(reg.coef_)
    beta = np.transpose(beta)
    beta0 = np.array(reg.intercept_)

    beta = np.vstack([beta0, beta])

    # printModelEquationLR(beta)

    # print(f'beta = {beta}')

    y_train_pred = modelAddOnesToX(x_train, beta)
    SSE_train = calculateSSE(y_train, y_train_pred)
    # print(f'Irrelevant: SSE_train = {SSE_train}')

    y_test_pred = modelAddOnesToX(x_test, beta)
    np.save('data/Ytest_pred_Regression1.npy', y_test_pred)
    
    # print(f'x_train.shape = {x_train.shape}')
    # print(f'y_train.shape = {y_train.shape}')
    # print(f'x_test.shape = {x_test.shape}')
    # print(f'y_test_pred.shape = {y_test_pred.shape}')
    
    if (y_test_pred.shape != (1000, 1)):
        print('confirm that both of your outputs have the same shape')
        
    # y_test_pred = np.load('data/Ytest_pred_Regression1.npy')
    # print(f'y_test_pred = {y_test_pred}')
    # print(f'y_test_pred.shape = {y_test_pred.shape}')
    

# Spliting the training data into random train and test subsets to perform a model performance evaluation on Ridge Regression
def RidgeTesting(test_size1):
    SSE = []
    alphas = list(range (1, 10000, 1))

    for alphaa in alphas:
        alphaa = alphaa * 0.0001
        x_train_original = np.load("data/Xtrain_Regression1.npy")
        y_train_original = np.load("data/Ytrain_Regression1.npy")

        x_train, x_test, y_train, y_test = train_test_split(x_train_original, y_train_original, test_size = test_size1, random_state=42)

        if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
            sys.exit()

        reg = linear_model.Ridge(alpha=alphaa)
        reg.fit(x_train, y_train)
        beta = np.array(reg.coef_)
        beta = np.transpose(beta)
        beta0 = np.array(reg.intercept_)
        beta = np.vstack([beta0, beta])

        y_test_pred = modelAddOnesToX(x_test, beta)

        SSE_test = calculateSSE(y_test, y_test_pred)
        # print(f'SSE_validation_data = {SSE_test} for alpha = {alphaa}')
        SSE.append(SSE_test)
        
    best_alpha = (SSE.index(min(SSE)) + 1) * 0.0001
    print(f'Best case: SSE_validation_data = {min(SSE)} for alpha = {best_alpha}')
    
    plt.figure(figsize=(5, 3))
    
    for i in range(len(alphas)): 
        alphas[i] = alphas[i] * 0.0001
    
    plt.plot(alphas, SSE)
    plt.grid()
    plt.xlim(0,1)
    plt.title('SSE validation data in function of Ridge´s alpha')
    
    plt.xlabel('alpha')
    plt.ylabel('SSE validation data')
    plt.tight_layout()
    plt.show()
    
# Spliting the training data into random train and test subsets to perform a model performance evaluation on Ridge Regression
def RidgeTestingkfolds(kk):
    alphas = list(range (1, 10000, 1))
    x_train_original = np.load("data/Xtrain_Regression1.npy")
    y_train_original = np.load("data/Ytrain_Regression1.npy")
    SSE = []
    scores_list = []
    
    for alphaa in alphas:
        scores_list.clear
        
        for i in range(2,12):
            alphaa = alphaa * 0.0001 
            # define the test condition
            cv = KFold(n_splits=kk, shuffle=True, random_state=i)
            # evaluate k value with the mean squared error
            scores = -cross_val_score(linear_model.Ridge(alpha = alphaa), x_train_original, y_train_original, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
            scores_list.append(scores)
        
        SSE_test = (x_train_original.shape[0]  / kk) * mean(scores_list)
        # print(f'SSE_validation_data = {SSE_test}')
        SSE.append(SSE_test)
        
    best_alpha = (SSE.index(min(SSE)) + 1) * 0.0001
    print(f'Best case: SSE_validation_data = {min(SSE)} for alpha = {best_alpha}')

# Predict y_train by Lasso Regression
def Lasso(alpha1):
    x_train = np.load("data/Xtrain_Regression1.npy")
    y_train = np.load("data/Ytrain_Regression1.npy")
    x_test = np.load("data/Xtest_Regression1.npy")

    if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
        sys.exit()
        
    reg = linear_model.Lasso(alpha=alpha1)
    reg.fit(x_train, y_train)

    beta = np.array([reg.coef_])
    beta = np.transpose(beta)
    
    beta0 = np.array([reg.intercept_])
    beta0 = np.transpose(beta0)
        
    y_train_pred = model(x_train, beta, beta0)
    SSE_train = calculateSSELasso(y_train, y_train_pred)
    print(f'Irrelevant: SSE_train = {SSE_train}')

    y_test_pred = model(x_test, beta, beta0)
    np.save('data/Ytest_pred_Regression1.npy', y_test_pred)
    
    # print(f'x_train.shape = {x_train.shape}')
    # print(f'y_train.shape = {y_train.shape}')
    # print(f'x_test.shape = {x_test.shape}')
    # print(f'y_test_pred.shape = {y_test_pred.shape}')
    if (y_test_pred.shape != (1000, 1)):
        print('confirm that both of your outputs have the same shape')

# Spliting the training data into random train and test subsets to perform a model performance evaluation on Lasso Regression
def LassoTesting(test_size1):
    
    alphas = list(range (1, 10000, 1))
    SSE = []

    for alphaa in alphas:
        alphaa = alphaa * 0.0001 
        x_train_original = np.load("data/Xtrain_Regression1.npy")
        y_train_original = np.load("data/Ytrain_Regression1.npy")

        x_train, x_test, y_train, y_test = train_test_split(x_train_original, y_train_original, test_size = test_size1, random_state=42)

        if (x_train.shape[0] != y_train.shape[0]) or (x_train.shape[0] <= 0):
            sys.exit()
            
        reg = linear_model.Lasso(alpha=alphaa)
        reg.fit(x_train, y_train)

        beta = np.array([reg.coef_])
        beta = np.transpose(beta)
        
        beta0 = np.array([reg.intercept_])
        beta0 = np.transpose(beta0)

        y_test_pred = model(x_test, beta, beta0)
        SSE_test = calculateSSELasso(y_test, y_test_pred)
        # print(f'SSE_validation_data = {SSE_test} for alpha = {alphaa}')
        SSE.append(SSE_test)
        
    best_alpha = (SSE.index(min(SSE)) + 1) * 0.0001
    print(f'Best case: SSE_validation_data = {min(SSE)} for alpha = {best_alpha}')
    
    plt.figure(figsize=(5, 3))
    
    for i in range(len(alphas)): 
        alphas[i] = alphas[i] * 0.0001
    
    plt.plot(alphas, SSE)
    plt.grid()
    plt.xlim(0,1)
    plt.title('SSE validation data in function of Lasso´s alpha')
    
    plt.xlabel('alpha')
    plt.ylabel('SSE validation data')
    plt.tight_layout()
    plt.show()

# Spliting the training data into random train and test subsets to perform a model performance evaluation on Ridge Regression
def LassoTestingkfolds(kk):
    alphas = list(range (1, 10000, 1))
    x_train_original = np.load("data/Xtrain_Regression1.npy")
    y_train_original = np.load("data/Ytrain_Regression1.npy")
    SSE = []
    scores_list = []
    
    for alphaa in alphas:
        scores_list.clear
        for i in range(2,5):
            alphaa = alphaa * 0.0001 
            # define the test condition
            cv = KFold(n_splits=kk, shuffle=True, random_state=i)
            # evaluate k value
            scores = -cross_val_score(linear_model.Lasso(alpha = alphaa), x_train_original, y_train_original, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
            scores_list.append(scores)
            
        SSE_test = (x_train_original.shape[0]  / kk) * mean(scores_list)
        # print(f'SSE_validation_data = {SSE_test}')
        SSE.append(SSE_test)
        
    best_alpha = (SSE.index(min(SSE)) + 1) * 0.0001
    print(f'Best case: SSE_validation_data = {min(SSE)} for alpha = {best_alpha}')
    

# We pick Ridge Regression with alpha = 0.2357 to apply to the test set and perform the submission

# print('Poly(degree = 2):')
# Poly(2)

# print('\nPolyTesting(degree = 2, test_size = 0.3)')
# PolyTesting(2, 0.3)

# print('\nPoly(degree = 3)')
# Poly(3)

# print('\nPolyTesting(degree = 3, test_size = 0.3)')
# PolyTesting(3, 0.3)

# print('\nPolykfolds(degree = 2, kk = 5)')
# Polykfolds(2, 5)

# print('\nPolykfolds(degree = 3, kk = 5)')
# Polykfolds(3, 5)

# print('\nLinRegressionMath()')
# LinRegressionMath()

#print('\nLinRegression()')
#LinRegression()

# print('\nLinRegressionTesting(test_size = 0.3)')
# LinRegressionTesting(0.3)

# print('\nLinRegressionTestingkfolds(k = 5)')
# LinRegressionTestingkfolds(5)

print('Ridge(alpha = 0.2357)')
Ridge(0.2357)

# print('\nRidgeTesting(test_size = 0.3)')
# RidgeTesting(0.3)

# print('\nRidgeTestingkfolds(k = 5)')
# RidgeTestingkfolds(5)

# print('\nLasso(alpha = 0.0024)')
# Lasso(0.0024)

# print('\nLassoTesting(test_size = 0.3)')
# LassoTesting(0.3)

# print('\nLassoTestingkfolds(k = 5)')
# LassoTestingkfolds(5)