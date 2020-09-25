# import standard module
import pdb
import numpy as np
from itertools import combinations

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

    """
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
    dp.get_DataMatrix()

    # define linear regression class
    total_fold = 5
    regressor = LinearRegression()
    regressor_reg = LinearRegression(reg=True, _lambda=0.7)
    
    for fold in range(total_fold):
        X_train, X_test, y_train, y_test = dp.split(total_fold, fold+1)
        # get weight
        weight = regressor.fit(X_train, y_train)
        weight_reg = regressor_reg.fit(X_train, y_train)
        
        # predict
        pred = regressor.predict(X_test)
        pred_reg = regressor_reg.predict(X_test)
        
        # compute RSS
        sqrt_RSS = regressor.RSS_error(y_test)
        sqrt_RSS_reg = regressor_reg.RSS_error(y_test)

        # plot the figure of prediction
        plot_line([pred, pred_reg, y_test], ["pred", "pred w/reg", "true"], ["sample", "combat point"])
        
        print('Fold {}: sqrt RSS = {}'.format(fold+1, sqrt_RSS))
        print('Fold {}: sqrt RSS with regularization = {}'.format(fold+1, sqrt_RSS_reg))

