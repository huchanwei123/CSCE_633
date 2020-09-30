import csv
import pdb
import math
import random
import numpy as np

class data_preprocess(object):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = []  # total data
        self.cat_map = {}
        self.num_attr = ['stamina', 'attack_value', 'defense_value', 'capture_rate', 'flee_rate', 'spawn_chance']
        self.cat_attr = ['primary_strength']
        self.len_one_hot = 0

    def read_csv(self):
        # read CSV file
        with open(self.csv_path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            cnt = 0
            for row in csv_reader:
                feat_dict = {}  # clean the dictionary
                if cnt == 0:
                    self.total_attr = row[1:len(row)-1]
                else:
                    feat_dict['name'] = row[0]
                    feat_dict['stamina'] = float(row[1])
                    feat_dict['attack_value'] = float(row[2])
                    feat_dict['defense_value'] = float(row[3])
                    feat_dict['capture_rate'] = float(row[4])
                    feat_dict['flee_rate'] = float(row[5])
                    feat_dict['spawn_chance'] = float(row[6])
                    feat_dict['primary_strength'] = row[7]
                    feat_dict['combat_point'] = float(row[8])
                    self.data.append(feat_dict)
                cnt = cnt + 1
        # shuffle data
        #random.shuffle(self.data)

    def one_hot_encoding(self):
        # get the categories
        cat = set([x['primary_strength'] for x in self.data])

        # sort the categories alphabetically to ensure the encoding won't change
        cat = sorted(cat, key=str.lower)
        self.len_one_hot = len(cat)

        # assign index to each category
        self.cat_map = dict((c, i) for i, c in enumerate(cat))

        # start encoding (iterative method -> stupid solution)
        one_hot = np.zeros((len(self.data), len(self.cat_map)))
        for i in range(len(self.data)):
            cur_cat = self.data[i]['primary_strength']
            one_hot[i][self.cat_map[cur_cat]] = 1
        return one_hot

    def get_feature(self, attribute):
        # get the numpy array of attribute for all data
        assert attribute != 'name'  # don't need the name of Pokemon
        if attribute == 'primary_strength':
            # using one hot encoding for categorical variables
            return self.one_hot_encoding()
        else:
            return np.asarray([x[attribute] for x in self.data])

    def get_DataMatrix(self, attr_list='all', binarize_outcome=False):
        """
        choose the attribute and get the data matrix
        Input:
            attr_list: attribute list you are interested
        Return:
            NULL
        """
        if attr_list == 'all':
            attr_list = self.total_attr

        # first combine all the feature to a matrix
        num_feature = len(self.data)
        if ('primary_strength' in attr_list):
            primary_strg = self.one_hot_encoding()
            feature_dim = self.len_one_hot + len(attr_list)
        else:
            feature_dim = 1 + len(attr_list)

        self.X = np.zeros((num_feature, feature_dim))
        self.y = np.zeros((num_feature, 1))

        for i in range(num_feature):
            feat_vec = np.array([1.0])
            for attr in attr_list:
                if attr == self.cat_attr[0]:
                    feat_vec = np.append(feat_vec, primary_strg[i])
                else:
                    feat_vec = np.append(feat_vec, self.data[i][attr])

            self.X[i] = feat_vec

            # check if we need to binarize outcome for logistic regression
            if not binarize_outcome:
                self.y[i] = self.data[i]['combat_point']
        
        if binarize_outcome:
            mean_outcome = sum(self.get_feature('combat_point')) / num_feature
            self.y = np.where(self.get_feature('combat_point') > mean_outcome, 1, 0)    
        return self.X, self.y

    def split(self, X, y, num_fold, fold_to_test):
        # start splitting
        fz = math.floor(len(X)/num_fold)

        if fold_to_test == num_fold:
            X_train = X[:fz*(fold_to_test-1)]
            y_train = y[:fz*(fold_to_test-1)]
            X_test = X[fz*(fold_to_test-1):]
            y_test = y[fz*(fold_to_test-1):]
        elif fold_to_test == 1:
            X_train = X[fz:]
            y_train = y[fz:]
            X_test = X[:fz]
            y_test = y[:fz]
        else:
            X_train = np.concatenate((X[:fz*(fold_to_test-1)], X[fz*fold_to_test:]), axis=0)
            y_train = np.concatenate((y[:fz*(fold_to_test-1)], y[fz*fold_to_test:]), axis=0)
            X_test = X[fz*(fold_to_test-1):fz*fold_to_test]
            y_test = y[fz*(fold_to_test-1):fz*fold_to_test]

        return X_train, X_test, y_train, y_test
