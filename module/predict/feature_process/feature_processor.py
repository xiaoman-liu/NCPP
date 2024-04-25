#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/22/2022 5:03 PM
# @Author  : xiaomanl
# @File    : feature.py
# @Software: PyCharm

import pandas as pd
import logging

from src.train.train_utils import calculate_running_time, dict_to_csv, generate_evaluate_metric
from src.train.model import XgboostModel
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class FeatureProcessor():
    def __init__(self, configs, feature, label, train_inds, test_inds, mapping):
        """
        Initialize the FeatureHandler class
        :param args: Command line arguments
        :param configs: Configuration data from the YAML file
        :param feature: DataFrame containing the features
        :param label: DataFrame containing the labels
        :param train_inds: Indices of the training data
        :param test_inds: Indices of the testing data
        """
        self.configs = configs
        self.logger = logging.getLogger("FeatureHandler")
        self.feature = self.feature_reduction(feature)
        self.label = label
        self.train_inds = train_inds
        self.test_inds = test_inds
        self.feature_mapping = mapping
        self.x_train = feature.loc[self.train_inds]
        self.x_test = feature.loc[self.test_inds]
        self.y_train = label.loc[self.train_inds]
        self.y_test = label.loc[self.test_inds]
        self.feature_group_feature = ['RESULT.WorkloadName', 'RESULT.TestName']



    def feature_reduction(self, feature):
        #reducted_feature = feature.drop(columns=["kubernetes.pod_id", "RESULT.kubernetes.host", "RESULT.cluster-name", ""])
        reducted_feature = feature
        return reducted_feature

    def train_data_processing(self, x_train, y_train):

        return x_train, y_train

    def merge_itereations(self, data):
        group_features = list(data.columns.difference(["RESULT.IterationIndex", "RESULT.Value"]))
        grouped_data = data.groupby(group_features)
        if grouped_data is None:
            self.logger.error(f"Data is NONE after filter, please check the {'filter_groups'} variable")
        data = grouped_data.mean().reset_index()

        return data


    def split_data(self, split=None):
        """
        Divide the features and labels into training and testing sets
        :return: Tuple containing the training and testing features and labels
        """
        if split is None:
            if self.configs["merge_iterations_value"]:
                data = pd.concat([self.feature.loc[self.train_inds],self.label.loc[self.train_inds]])
                merge_data = self.merge_itereations(data)
                x_train, y_train = merge_data[list(merge_data.columns.difference(["RESULT.Value"]))], merge_data[["RESULT.Value"]]
            else:
                x_train, y_train = self.feature.loc[self.train_inds], self.label.loc[self.train_inds]
            x_test, y_test = self.feature.iloc[self.test_inds], self.label.iloc[self.test_inds]
        else:
            if self.configs["merge_iterations_value"]:
                data = pd.concat([self.feature.loc[self.train_inds[split]], self.label.loc[self.train_inds[split]]])
                merge_data = self.merge_itereations(data)
                x_train, y_train = merge_data[list(merge_data.columns.difference(["RESULT.Value"]))], merge_data[
                    ["RESULT.Value"]]
            else:
                x_train, y_train = self.feature.loc[self.train_inds[split]], self.label.loc[self.train_inds[split]]
            x_test, y_test = self.feature.loc[self.test_inds[split]], self.label.loc[self.test_inds[split]]

        self.logger.info("Finish splitting data")
        self.x_train,self.y_train = self.train_data_processing(x_train.reset_index(drop=True),y_train.reset_index(drop=True) )
        self.y_test = y_test.reset_index(drop=True)
        self.x_test = x_test.reset_index(drop=True)



        return self.x_train, self.y_train, self.x_test, self.y_test

    def feature_ranking_rf(self):
        """
        Rank features using random forest
        :param X: feature matrix
        :param y: target vector
        :return: feature importance scores
        """
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(self.x_train, self.y_train)
        # importances = rf.feature_importances_
        # indices = np.argsort(importances)[::-1]
        feature_set = set()
        for type in "gain":
            imp = rf.get_booster().get_score(importance_type=type)
            # feature_imp_sorted = pd.DataFrame(imp)
            imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
            dict_to_csv(imp, type, self.configs["save_path"])
            header = set(list(imp.keys())[:50])
            if not feature_set:
                feature_set.update(header)
            else:
                feature_set = feature_set & header
        return

    def feature_ranking_xgb(self):
        """
        Rank features using XGBoost
        :param X: feature matrix
        :param y: target vector
        :return: feature importance scores
        """
        xgb_model = xgb.XGBRegressor(
            max_depth=50,
            n_estimators=300,
            learning_rate=0.1,
            verbosity=1,
            booster="gbtree",
            n_jobs=10,
            random_state=1,
        )
        xgb_model.fit(self.x_train, self.y_train)
        # importances = xgb_model.feature_importances_
        # indices = np.argsort(importances)[::-1]
        feature_set = set()
        for type in ["gain"]:
            imp = xgb_model.get_booster().get_score(importance_type=type)
            # feature_imp_sorted = pd.DataFrame(imp)
            imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
            dict_to_csv(self.configs, imp, type, self.configs["save_path"])
            header = set(list(imp.keys())[:50])
            if not feature_set:
                feature_set.update(header)
            else:
                feature_set = feature_set & header
        #return indices, importances
    def data_split(self, filter_data, all_label):
        if self.configs["merge_iterations_value"]:

            if self.configs["if_K_fold"]:
                single_kfold = StratifiedKFold(n_splits=self.configs["n_split"])
                all_train_indices = []
                all_test_indices = []
                for train_index, val_index in single_kfold.split(filter_data, all_label):
                    all_train_indices.append(train_index)
                    all_test_indices.append(val_index)
            else:
                splitter = ShuffleSplit(train_size=0.8, n_splits=5,
                                        random_state=0)
                all_train_indices, all_test_indices = next(splitter.split(filter_data, all_label))
        else:
            groups = filter_data[["kubernetes.pod_id", "RESULT.WorkloadName", "RESULT.TestName"]].apply(
                lambda x: '_'.join(x.astype(str)), axis=1)
            if self.configs["if_K_fold"]:

                # Initialize the GroupKFold class with the number of folds you want
                group_kfold = GroupKFold(n_splits=self.configs["n_split"])
                # Split the data into train and test sets using the split method
                # all_train_indices = np.array([])
                # all_test_indices = np.array([])
                # for train_index, val_index in group_kfold.split(filter_data, all_label, groups):
                #     all_train_indices = np.concatenate([all_train_indices, train_index])
                #     all_test_indices = np.concatenate([all_test_indices, val_index])
                all_train_indices = []
                all_test_indices = []
                for train_index, val_index in group_kfold.split(filter_data, all_label, groups):
                    all_train_indices.append(train_index)
                    all_test_indices.append(val_index)
            else:
                splitter = GroupShuffleSplit(train_size=0.8, n_splits=5,
                                             random_state=2021)

                # Train and evaluate your model on the training and validation sets
                all_train_indices, all_test_indices = next(splitter.split(filter_data, all_label, groups=groups))

            # groups = filter_data[["kubernetes.pod_id"]].apply(lambda x: '_'.join(x.astype(str)), axis=1)
            #
        return all_train_indices, all_test_indices
    def feature_ranking_xgb2(self, i=1000):
        """
        Rank features using XGBoost
        :param X: feature matrix
        :param y: target vector
        :return: feature importance scores
        """

        # Create the dataset

        data = pd.concat([self.x_train, self.y_train], axis =1)
        grouped_data = data.groupby(self.feature_group_feature)

        # Initialize the result dataframe
        evaluate_header= ['Count','MAPE', 'P90_APE(%)', 'P95_APE(%)', 'P99_APE(%)', 'MAX_APE(%)', 'ACC(APE<3%)']
        headers = ['RESULT.WorkloadName'] + ['RESULT.TestName'] + evaluate_header +[f"rank_{i}" for i in range(1, 41)]
        result = pd.DataFrame(columns=headers)

        # Loop through each group of data
        feature_set = set()
        variation_summary = pd.DataFrame()
        for name, group in grouped_data:
            workloadname, testname = name
            mapped_workloadname = next(key for key, val in self.feature_mapping["RESULT.WorkloadName"].items() if val == workloadname)
            mapped_testname = next(key for key, val in self.feature_mapping["RESULT.TestName"].items() if val == testname)


            # Prepare the data for XGBoost
            y = group[["RESULT.Value"]]
            X = group.drop(["RESULT.Value"], axis=1)

            # Train the XGBoost model
            A = XgboostModel()
            model = A.build_model()
            model.fit(X, y)
            get_test_rule = self.x_test["RESULT.TestName"] == testname
            each_case_x_test = self.x_test[get_test_rule].reset_index(drop=True)
            predict = pd.DataFrame(model.predict(each_case_x_test), columns=["y_predict"])


            # Plot the feature importance
            #plot_importance(model)

            # Get the feature importance ranking

            feature_importance = model.get_booster().get_score(importance_type='gain')
            total = sum(feature_importance.values())
            feature_importance = {key: round(val / total * 100, 3) for key, val in feature_importance.items()}

            # Sort the feature importance in descending order
            sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            header = set([key for key, value in feature_importance.items() if value > 0.5])
            if not feature_set:
                feature_set.update(header)
            else:
                feature_set = feature_set.union(header)

            # Count the top 10 feature ranking
            top_10_features = sorted_feature_importance


            # Add the top 10 feature ranking to the result dataframe
            row = [mapped_workloadname, mapped_testname]
            true_value = self.y_test[get_test_rule].reset_index(drop=True)
            evaluate_metric, compare = generate_evaluate_metric(predict, true_value)
            row += evaluate_metric
            for j, (feature, score) in enumerate(top_10_features):
                # row.append(f'{feature} : {score}')
                row.append(feature)
            result = result.append(dict(zip(headers, row)), ignore_index=True)
            variation_data = pd.concat([each_case_x_test, compare], axis=1)
            filtered_variation = variation_data[variation_data['APE'] > 10]
            variation_summary = pd.concat([filtered_variation, variation_summary])
            #

        # Save the result dataframe to a CSV file
        variation_data_result = variation_summary.reset_index(drop=True)
        # variation_data_result = pd.DataFrame()

        for col in variation_summary.columns:
            if col in self.feature_mapping:
                mapping = self.feature_mapping[col]
            else:
                continue
            variation_data_result[col] = [k for v in variation_summary[col] for k, val in mapping.items() if val == v]
        # variation_summary = variation_summary.applymap(lambda x: next((k for k, v in self.feature_mapping[col].items() if v == x), None) for col, x in variation_summary.stack().items()).unstack()
        # variation_data_result.to_csv(f"{self.configs['save_path']}/split_{i}/variation_summary.csv", index=False)




        feature_list = pd.DataFrame({"feature_list":list(feature_set)})
        all_feature_set = set(list(data.columns))
        slience_header = pd.DataFrame(all_feature_set - feature_set)
        slience_header.to_csv(f'{self.configs["output_path"]}/slience_header.csv', index=False, header=False)
        result.to_csv(f'{self.configs["output_path"]}//feature_ranking.csv', index=False)
        counts = result[[f"rank_{i}" for i in range(1, 41)]].apply(pd.Series.value_counts)
        counts.to_csv(f'{self.configs["output_path"]}/feature_ranking_count.csv')
        feature_list.to_csv(f'{self.configs["output_path"]}/feature_list.csv', index=False)
        self.logger.info(f"Feature ranking results are saving to {self.configs['output_path']}/feature_ranking.csv")

    @calculate_running_time
    def run(self):
        """
        Handle the features
        :return: Tuple containing the training and testing features and labels
        """
        x=0
        if x:
            for i in range(int(self.configs["n_split"])):
                # self.split_data(split=i)
                if self.configs["feature_ranking"]:
                    self.feature_ranking_xgb2(i)

                    self.logger.info(f"split {i}: Use XgboostModel for feature ranking.")

                # self.feature_ranking_rf()
                self.logger.info("######"*4)
                self.logger.info(
                    f"split {i}: The shape of x_train is: \033[1;34;34m{self.x_train.shape}")
                self.logger.info(
                    f"split {i}: The shape of y_train is: \033[1;34;34m{self.y_train.shape}")
                self.logger.info(
                    f"split {i}: The shape of x_test is: \033[1;34;34m{self.x_test.shape}")
                self.logger.info(
                    f"split {i}: The shape of y_test is: \033[1;34;34m{self.y_test.shape}")
        else:
            # self.split_data()
            if self.configs["feature_ranking"]:
                self.feature_ranking_xgb2()

                self.logger.info(f"Use XgboostModel for feature ranking.")
            # self.logger.info(
            #     f"The shape of x_train is: \033[1;34;34m{self.x_train.shape}")
            # self.logger.info(
            #     f"The shape of y_train is: \033[1;34;34m{self.y_train.shape}")
            # self.logger.info(
            #     f"The shape of x_test is: \033[1;34;34m{self.x_test.shape}")
            # self.logger.info(
            #     f"The shape of y_test is: \033[1;34;34m{self.y_test.shape}")

        return self.x_train, self.y_train, self.x_test, self.y_test


