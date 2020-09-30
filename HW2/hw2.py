# import standard module
import pdb
import numpy as np
from itertools import combinations
from sklearn import metrics
from sklearn.linear_model import LogisticRegression as Log_R

# import my own module
from my_data_preprocess import data_preprocess
from my_misc import compute_Pearson, plot_2D, plot_line
from my_LR import LinearRegression

if __name__ == '__main__':
    # preproces data first
    dp = data_preprocess('./hw2_data.csv')
    dp.read_csv()
    numerical_attr = ['stamina', 'attack_value', 'defense_value', 'capture_rate', 'flee_rate', 'spawn_chance']
    categorical_attr = 'primary_strength'
    cpo = 'combat_point'    # combat point outcome

    """ 
    # Question (ii)
    corr = -np.Inf
    min_corr_attr = ''
    tmp = 0
    attr2 = dp.get_feature(cpo)
    for attr in numerical_attr:
        attr1 = dp.get_feature(attr)
        tmp = compute_Pearson(attr1, attr2)
        if tmp > corr:
            min_corr_attr = attr
            corr = tmp
        plot_2D(attr1, attr, attr2, cpo)
        print('Pearson correlation coefficient of {} and {} is {}'.format(attr, cpo, tmp))
    print('Most predictive outcome of combat points is {}, Pearson correlation coefficient is {}'.format(min_corr_attr, corr))

    
    # Question (iii)
    attribute_comb = list(combinations(numerical_attr, 2))
    for c in attribute_comb:
        attr1 = dp.get_feature(c[0])
        attr2 = dp.get_feature(c[1])
        Pcoef = compute_Pearson(attr1, attr2)
        plot_2D(attr1, c[0], attr2, c[1])
        print('Pearson correlation coefficient of {} and {} is {}'.format(c[0], c[1], Pcoef))

    """
    # Question (v)
    X_all, y_all = dp.get_DataMatrix()

    # define linear regression class
    total_fold = 5
    
    """
    for fold in range(total_fold):
        X_train, X_test, y_train, y_test = dp.split(X_all, y_all, total_fold, fold+1)
        # get weight
        weight = regressor.fit(X_train, y_train)
        weight_reg = regressor_reg.fit(X_train, y_train)
        
        # predict
        pred = regressor.predict(X_test)
        pred_reg = regressor_reg.predict(X_test)
        
        # compute RSS
        sqrt_RSS = regressor.RSS_error(y_test)
        sqrt_RSS_reg = regressor_reg.RSS_error(y_test)

        sqrt_reg_sum += sqrt_RSS_reg
        sqrt_sum += sqrt_RSS
        # plot the figure of prediction
        #plot_line([pred, pred_reg, y_test], ["pred", "pred w/reg", "true"], ["Pokeman", "combat point"]) 
        plot_line([pred, y_test], ["predict", "true outcome"], ["Pokeman", "combat point"])

        print('Fold {}: sqrt RSS = {}'.format(fold+1, sqrt_RSS))
        
    print('Average sqrt RSS = {}'.format(sqrt_sum/total_fold))
    """

    # Question (vi)
    print('\n----------Question (vi)+(vii)----------')
    lambda_list = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 10]
    
    for _lambda in lambda_list:
        regressor = LinearRegression(reg=True, _lambda=_lambda)
        
        sqrt_sum = 0
        for fold in range(total_fold):
            X_train, X_test, y_train, y_test = dp.split(X_all, y_all, total_fold, fold+1)
            # get weight
            weight = regressor.fit(X_train, y_train)

            # predict
            pred = regressor.predict(X_test)

            # compute RSS
            sqrt_RSS = regressor.RSS_error(y_test)

            # store the error over folds
            sqrt_sum += sqrt_RSS
            
            # plot the figure of prediction
            #plot_line([pred, pred_reg, y_test], ["predcit", "predict w/reg", "true outcome"], ["Pokeman", "combat point"])

            print('Fold {}: sqrt RSS = {}'.format(fold+1, sqrt_RSS))
        
        if _lambda == 0:
            print('Average sqrt RSS = {}'.format(sqrt_sum/total_fold))
        else:
            print('Average sqrt RSS with regularization {} = {}'.format(_lambda, sqrt_sum/total_fold))

    
    # Question (viii)
    print('\n----------Question (viii)----------')
    X_all, y_all = dp.get_DataMatrix(binarize_outcome = True)
    
    # split 80-20
    X_train, X_test, y_train, y_test = dp.split(X_all, y_all, 5, 1)

    logistic_regression = Log_R(penalty='none', solver='lbfgs', max_iter=10000)
    logistic_regression.fit(X_train, y_train)
    pred = logistic_regression.predict(X_test)
    # calculate accuracy
    accu = metrics.accuracy_score(y_test, pred) * 100

    print('Accuracy of logistic regression without regularization = {}'.format(accu))

    print('\n----------Question (ix)----------')
    # Logistic regression with regularization
    lambda_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    avg_accu = []
    for _lambda in lambda_list:
        # 5-fold cross validation
        sum_accu = 0
        for f in range(total_fold):
            X_train_, X_test_, y_train_, y_test_ = dp.split(X_train, y_train, total_fold, f+1)
            logistic_regression = Log_R(C=1.0/_lambda, penalty='l2', solver='lbfgs', max_iter=10000)
            logistic_regression.fit(X_train_, y_train_)
            pred = logistic_regression.predict(X_test_)
            accu = metrics.accuracy_score(y_test_, pred) * 100
            sum_accu += accu
        print("Average accuracy of logistic regression with lambda = {} is {}".format(_lambda, sum_accu/total_fold))
        avg_accu.append(sum_accu/total_fold)

    # choose the optimal lambda
    lambda_best = lambda_list[avg_accu.index(max(avg_accu))]
    logistic_regression = Log_R(C=1.0/lambda_best, penalty='l2', solver='lbfgs', max_iter=10000)
    logistic_regression.fit(X_train, y_train)
    pred = logistic_regression.predict(X_test)
    accu = metrics.accuracy_score(y_test, pred) * 100

    print('Accuracy of logistic regression with lambda = {} is {}'.format(lambda_best, accu))
    
