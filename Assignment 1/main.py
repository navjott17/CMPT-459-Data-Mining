import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from random_forest import RandomForest

def parse_args():
    parser = argparse.ArgumentParser(description='Run random forrest with specified input arguments')
    parser.add_argument('--n-classifiers', type=int,
                        help='number of features to use in a tree',
                        default=1)
    parser.add_argument('--train-data', type=str, default='data/train.csv',
                        help='train data path')
    parser.add_argument('--test-data', type=str, default='data/test.csv',
                        help='test data path')
    parser.add_argument('--criterion', type=str, default='entropy',
                        help='criterion to use to split nodes. Should be either gini or entropy.')
    parser.add_argument('--maxdepth', type=int, help='maximum depth of the tree',
                        default=5)
    parser.add_argument('--min-sample-split', type=int, help='The minimum number of samples required to be at a leaf node',
                        default=20)
    parser.add_argument('--max-features', type=int,
                        help='number of features to use in a tree',
                        default=11)
    a = parser.parse_args()
    return(a.n_classifiers, a.train_data, a.test_data, a.criterion, a.maxdepth, a.min_sample_split, a.max_features)


def read_data(path):
    data = pd.read_csv(path)
    return data

def data_clean(data):
    data = data.replace(' ?', np.nan)
    data = data.dropna()
    data.drop('fnlwgt', axis = 1, inplace = True)
    data['income'] = data.income.str.replace(".", "", regex=True)
    return data

def main():
    n_classifiers, train_data_path, test_data_path, criterion, max_depth, min_sample_split, max_features = parse_args()
    train_data = read_data(train_data_path)
    test_data = read_data(test_data_path)
    # YOU NEED TO HANDLE MISSING VALUES HERE
    # ...
    #data cleaning
    train_data = data_clean(train_data)
    test_data = data_clean(test_data)

#     random_forest = RandomForest(n_classifiers=n_classifiers,
#                   criterion = criterion,
#                   max_depth=  max_depth,
#                   min_samples_split = min_sample_split ,
#                   max_features = max_features )
    
#     features = random_forest.process_features(train_data, 'income')
    
    1.
    random_forest = RandomForest(n_classifiers=10,
                  criterion = 'gini',
                  max_depth=  10,
                  min_samples_split = 20 ,
                  max_features = 11 )
    print(criterion)
    print(random_forest.fit(train_data, 'income'))
    # print(random_forest.evaluate(train_data, 'income'))
    print(random_forest.evaluate(test_data, 'income'))
    
    # 2. 
    # random_forest = RandomForest(n_classifiers=10,
    #               criterion = 'entropy',
    #               max_depth=  10,
    #               min_samples_split = 20 ,
    #               max_features = 11 )
    # print(criterion)
    # print(random_forest.fit(train_data, 'income'))
    # # print(random_forest.evaluate(train_data, 'income'))
    # print(random_forest.evaluate(test_data, 'income'))
    
    # 4.
    # plot_samples = []
    # for i in range(10):
    #     print("----------Depth ", i+1, "----------")
    #     random_forest = RandomForest(n_classifiers=10,
    #               criterion = 'gini',
    #               max_depth=  i+1,
    #               min_samples_split = 10 ,
    #               max_features = 10 )
    #     print("train: ", random_forest.fit(train_data, 'income'))
    #     plot_samples.append(random_forest.evaluate(test_data, 'income'))
    #     print("test: ", plot_samples[i])
    # fig = plt.figure(figsize=(5, 4))
    # plt.plot(np.linspace(1, 10, 10), plot_samples)
    # plt.xlabel('test_data_depth')
    # plt.ylabel('test_data_accuracy')
    # plt.savefig('accuracy.jpg')
    


if __name__ == '__main__':
    main()

