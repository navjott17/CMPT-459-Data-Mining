from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
import random


class Node(object):
    def __init__(self, node_size: int, node_class: str, depth: int, single_class:bool = False):
        # Every node is a leaf unless you set its 'children'
        self.is_leaf = True
        # Each 'decision node' has a name. It should be the feature name
        self.name = None
        # All children of a 'decision node'. Note that only decision nodes have children
        self.children = {}
        # Whether corresponding feature of this node is numerical or not. Only for decision nodes.
        self.is_numerical = None
        # Threshold value for numerical decision nodes. If the value of a specific data is greater than this threshold,
        # it falls under the 'ge' child. Other than that it goes under 'l'. Please check the implementation of
        # get_child_node for a better understanding.
        self.threshold = None
        # The class of a node. It determines the class of the data in this node. In this assignment it should be set as
        # the mode of the classes of data in this node.
        self.node_class = node_class
        # Number of data samples in this node
        self.size = node_size
        # Depth of a node
        self.depth = depth
        # Boolean variable indicating if all the data of this node belongs to only one class. This is condition that you
        # want to be aware of so you stop expanding the tree.
        self.single_class = single_class

    def set_children(self, children):
        self.is_leaf = False
        self.children = children

    def get_child_node(self, feature_value)-> 'Node':
        if not self.is_numerical:
            return self.children[feature_value]
        else:
            if feature_value >= self.threshold:
                return self.children['ge'] # ge stands for greater equal
            else:
                return self.children['l'] # l stands for less than


class RandomForest(object):
    def __init__(self, n_classifiers: int,
                 criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None,
                 max_features: Optional[int] = None):
        """
        :param n_classifiers:
            number of trees to generated in the forrest
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the trees.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        :param max_features:
            The number of features to consider for each tree.
        """
        self.n_classifiers = n_classifiers
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini
    
    # bagging (Bootstrap Aggregation) used in fit function where data is randomly selectly with replacement.
    def bagging(self, X: pd.DataFrame, y_col: str)->pd.DataFrame:
        random.random()
        temp = np.arange(X.shape[0])
        bag = np.random.choice(temp, X.shape[0])
        bagging_sample = X.iloc[bag]
        return bagging_sample
    
    # used bagging function and generate tree with replacement of data
    def fit(self, X: pd.DataFrame, y_col: str)->float:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of training dataset
        """
        self.trees.clear()
        features = self.process_features(X, y_col)
        # Your code  
        num_trees = self.n_classifiers
        for i in range(num_trees):
            data = self.bagging(X, y_col);
            tree_root = self.generate_tree(data, y_col, features)
            self.trees.append(tree_root)
        return self.evaluate(X, y_col)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        :param X: data
        :return: aggregated predictions of all trees on X. Use voting mechanism for aggregation.
        """
        votes_df = pd.DataFrame(columns = ['indx', 'pred'])
        for i in range(self.n_classifiers):
            root = self.trees[i]
            for j in range(X.shape[0]):
                while not root.is_leaf:
                    att = X.iloc[j][root.name]
                    if root.is_numerical:
                        root = root.get_child_node(att)
                    else:
                        if att in root.children:
                            root = root.get_child_node(att)
                        else:
                            break
                data = {'indx': [j], 'pred': root.node_class}
                tmp = pd.DataFrame.from_dict(data)
                votes_df = pd.concat([votes_df,tmp],ignore_index=True)
            
        votes_df = votes_df.groupby('indx').max()
        return np.array(votes_df.pred)

    def evaluate(self, X: pd.DataFrame, y_col: str)-> int:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == X[y_col]) / len(preds)
        return acc

    def generate_tree(self, X: pd.DataFrame, y_col: str, features: Sequence[Mapping])->Node:
        """
        Method to generate a decision tree. This method uses self.split_tree() method to split a node.
        :param X:
        :param y_col:
        :param features:
        :return: root of the tree
        """
        random.random()
        root = Node(X.shape[0], X[y_col].mode(), 0)
        self.split_node(root, X, y_col, features)
        return root
    
    def split_node(self, node: Node, X: pd.DataFrame, y_col:str, features: Sequence[Mapping]) -> None:
        """
        This is probably the most important function you will implement. This function takes a node, uses criterion to
        find the best feature to slit it, and splits it into child nodes. I recommend to use revursive programming to
        implement this function but you are of course free to take any programming approach you want to implement it.
        :param node:
        :param X:
        :param y_col:
        :param features:
        :return:
        """
        # base case check
        if X.shape[0] < self.min_samples_split or node.depth >= self.max_depth:
            return
        
        #Feature Randomness
        rand_features = random.sample(features,self.max_features)
        
        #chosing best entropy
        scores_list = []
        for i in range(len(rand_features)):
            crit_score = self.criterion_func(X, rand_features[i], y_col)
            scores_list.append(crit_score)
            
        if self.criterion == "gini":
            score = min(scores_list)
        else:
            score = max(scores_list)
        index = scores_list.index(score)
        
        #feature to be splitted found by splitting criterion
        min_score_feature = rand_features[index]['name']
        
        #check and set if the feature belongs to the single class
        if score == 0:
            node.single_class = True
            return
        else:
            node.single_class = False
        
        #check and set if the feature is numerical
        if rand_features[index]['dtype'] == object:
            node.is_numerical = False
        else:
            node.is_numerical = True
            
        node.name = min_score_feature
        feature_name = X[node.name]
        depth = node.depth
        
        #for categorical feature
        if node.is_numerical == False:
            unique_categories = feature_name.unique()
            for i in range(unique_categories.shape[0]):
                split = X[feature_name == unique_categories[i]]
                split_size = split.shape[0]
                y = split[y_col]
                node_new = Node(split_size, y.mode(), depth+1)
                self.split_node(node_new, split, y_col, features)
                node.children[unique_categories[i]] = node_new
        else:
            threshold = np.mean(feature_name)
            node.threshold = threshold
            split1 = X[feature_name < threshold]
            y1 = split1[y_col]
            split2 = X[feature_name >= threshold]
            y2 = split2[y_col]
            s1 = Node(split1.shape[0], y1.mode(), depth+1)
            self.split_node(s1, split1, y_col, features)
            
            s2 = Node(split2.shape[0], y2.mode(), depth+1)
            self.split_node(s2, split2, y_col, features)
            
            node.children['l'] = s1
            node.children['ge'] = s2
            
        if bool(node.children):
            node.is_leaf = False
        else:
            node.is_leaf = True
                 
        return
        


    def gini(self, X: pd.DataFrame, feature: Mapping, y_col: str) -> float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        gini_index = 0
        feature_name = X[feature['name']]
        y_unique = X[y_col].unique()
        
        # for categorical data
        if  X[feature['name']].dtype == object:
            unique_categories = feature_name.unique()
            for i in range(unique_categories.shape[0]):
                split = X[feature_name == unique_categories[i]]
                g_idx = 1
                if split.shape[0] > 0:
                    for j in range(y_unique.shape[0]):
                        similar_y = split[split[y_col] == y_unique[j]]
                        g_idx = g_idx - (similar_y.shape[0]/split.shape[0])*(similar_y.shape[0]/split.shape[0])
                gini_index = gini_index + (split.shape[0]/X.shape[0])*g_idx
        # for numerical data
        else:
            threshold = np.sum(feature_name)/feature_name.shape[0]
            split_1 = X[feature_name < threshold]
            split_2 = X[feature_name >= threshold]
            g_idx1 = 1
            g_idx2 = 1
            if split_1.shape[0] > 0:
                for i in range(y_unique.shape[0]):
                    similar_y = split_1[split_1[y_col] == y_unique[i]]
                    g_idx1 = g_idx1 - (similar_y.shape[0]/split_1.shape[0])*(similar_y.shape[0]/split_1.shape[0])
            if split_2.shape[0] > 0:
                for i in range(y_unique.shape[0]):
                    similar_y = split_2[split_2[y_col] == y_unique[i]]
                    g_idx2 = g_idx2 - (similar_y.shape[0]/split_2.shape[0])*(similar_y.shape[0]/split_2.shape[0])
            gini_index = (split_1.shape[0]/X.shape[0])*g_idx1 + (split_2.shape[0]/X.shape[0])*g_idx2
            
        return gini_index

    def entropy(self, X: pd.DataFrame, feature: Mapping, y_col: str) ->float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        entropy_parent = 0
        entropy = 0
        feature_name = X[feature['name']]
        y_unique = X[y_col].unique()
        
        for i in range(y_unique.shape[0]):
            similar_y = X[X[y_col] == y_unique[i]]
            prob = similar_y.shape[0]/X.shape[0]
            if prob == 0:
                entropy_parent = entropy_parent
                continue;
            entropy_parent = entropy_parent + (prob)*np.log2(prob)
        entropy_parent = -entropy_parent
            
        # for categorical data
        if feature_name.dtype == object:
            unique_categories = feature_name.unique()
            for i in range(unique_categories.shape[0]):
                split = X[feature_name == unique_categories[i]]
                ent = 0
                if split.shape[0] > 0:
                    for j in range(y_unique.shape[0]):
                        similar_y = split[split[y_col] == y_unique[j]]
                        prob = similar_y.shape[0]/split.shape[0]
                        if prob == 0:
                            ent = ent
                            continue;
                        ent = ent + prob*np.log2(prob)
                    entropy = entropy + ent*(split.shape[0]/X.shape[0])
        # for numerical features
        else:
            threshold = np.sum(feature_name)/feature_name.shape[0]
            split_1 = X[feature_name < threshold]
            split_2 = X[feature_name >= threshold]
            ent1 = 0
            ent2 = 0
            if split_1.shape[0] > 0:
                for i in range(y_unique.shape[0]):
                    similar_y = split_1[split_1[y_col] == y_unique[i]]
                    prob = similar_y.shape[0]/split_1.shape[0]
                    if prob == 0:
                        ent1 = ent1
                        continue;
                    ent1 = ent1 + prob*np.log2(prob)
            if split_2.shape[0] > 0:
                for i in range(y_unique.shape[0]):
                    similar_y = split_2[split_2[y_col] == y_unique[i]]
                    prob = similar_y.shape[0]/split_2.shape[0]
                    if prob == 0:
                        ent2 = ent2
                        continue;
                    ent2 = ent2 + prob*np.log2(prob)
            entropy = ent1*(split_1.shape[0]/X.shape[0]) + ent2*(split_2.shape[0]/X.shape[0])               
        entropy = -entropy
        info_gain = entropy_parent - entropy
        
        return info_gain

    def process_features(self, X: pd.DataFrame, y_col: str)->Sequence[Mapping]:
        """
        :param X: data
        :param y_col: name of the label column in X
        :return:
        """
        features = []
        for n,t in X.dtypes.items():
            if n == y_col:
                continue
            f = {'name': n, 'dtype': t}
            features.append(f)
        return features