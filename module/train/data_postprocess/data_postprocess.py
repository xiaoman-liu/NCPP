#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/13/2023 10:28 PM
# @Author  : xiaomanl
# @File    : data_postprocess.py
# @Software: PyCharm

import pandas as pd
import logging
import os
import numpy as np
from module.train.train_utils import calculate_running_time
import yaml
import shutil
from tabulate import tabulate


class DataPostprocessor():
    def __init__(self, final_save_path, configs, filtered_features, label, train_inds, test_inds, predicted_results, processed_data=None, hist_all=None):

        self.configs = configs
        self.config_save_path = self.configs["config_save_path"]
        self.output_path = self.configs["output_path"]
        self.final_save_path = final_save_path if final_save_path is not None else self.output_path
        self.logger = logging.getLogger("DataPostprocessor")
        self.label = label
        self.x_all_raw_data = filtered_features
        self.train_inds = train_inds
        self.test_inds = test_inds
        self.predicted_results = predicted_results
        self.label_name_list = self.configs["label_name_list"]
        self.predict_col = self.configs["predict_col"]
        self.true_col = self.configs["true_col"]
        self.workload_name = self.configs["workload_names"][0]
        self.label_scale = self.configs["label_scale"][self.workload_name]
        self.if_label_scale = self.configs["if_label_scale"]
        self.processed_data = processed_data
        self.train_with_all_data = self.configs["train_with_all_data"]
        self.hist_all_list = hist_all
        self.encoder_path = configs["encoder_path"]




    # def feature_map(self, discrete_raw_data):
    #     if self.configs["string_encoding"] == "Label_encoding":
    #         for col in discrete_raw_data.columns:
    #             discrete_raw_data[col] = discrete_raw_data[[col]].applymap(lambda x: next((key for key, val in self.feature_mapping[col].items() if val == x), x))
    #     elif self.configs["string_encoding"] == "Onehot_encoding":
    #         discrete_raw_data = discrete_raw_data.idxmax(axis=1)
    #
    #     return discrete_raw_data
    def save_data(self, data, file_name):
        data_path = os.path.join(self.final_save_path, file_name).replace("\\", "/")
        data.to_csv(data_path, index=False)
        self.logger.info("Save data to: {}".format(data_path))

    def overall_metric(self, label_col_name_list, predict_col_name_list):
        new_list = [x.split("_", 1)[1] for x in label_col_name_list]
        summary_data = pd.DataFrame(columns=["metric"] + new_list)
        summary_data["metric"] = ["MAE", "MSE", "MAPE(%)",
                                  "Accuracy (APE < 3%)", "Accuracy (APE < 5%)", "Accuracy (APE < 10%)",
                                  "P90 APE(%)", "P95 APE(%)", "P99 APE(%)", "MAX APE(%)"]
        for i in range(len(label_col_name_list)):
            label_col_name = label_col_name_list[i]
            predict_col_name = predict_col_name_list[i]
            origin_name = label_col_name.split("_", 1)[1]
            # Calculate the Absolute Error (AE) for dataframe
            y_test = self.predicted_results[label_col_name]
            y_predict = self.predicted_results[predict_col_name]
            plot_train_predict(y_test, y_predict, os.path.join(self.final_save_path, f"{origin_name}_.png"),
                                    title=f"{origin_name}: true value vs predict value")


            ae = abs(y_test - y_predict)
            mae = np.mean(ae)
            # Calculate the Mean Absolute Error (MAE)
            se = (y_test - y_predict) ** 2
            mse = np.mean(se)
            # Calculate the Mean Absolute Percentage Error (MAPE)
            ape = abs(y_test - y_predict) / y_test * 100


            mape = np.mean(ape)
            p50_ape = round(ape.quantile(0.5), 4)
            p90_ape = round(ape.quantile(0.9), 4)
            p95_ape = round(ape.quantile(0.95), 4)
            p99_ape = round(ape.quantile(0.99), 4)
            max_ape = round(np.max(ape), 4)
            count_3 = (ape < 3).sum()
            proportion_3 = round(count_3 / len(ape) * 100, 5)

            count_5 = (ape < 5).sum()
            proportion_5 = round(count_5 / len(ape) * 100, 4)
            count_10 = (ape < 10).sum()
            proportion_10 = round(count_10 / len(ape) * 100, 4)
            summary_data[origin_name] = [mae, mse, mape,
                                         proportion_3, proportion_5, proportion_10,
                                         p90_ape, p95_ape, p99_ape, max_ape]
            # self.logger.info(f"---------------------------{self.configs['k_fold_order'] } Fold metrics snippet---------------------")
            # self.logger.info(f"MAE: {mae:.4f}")
            # self.logger.info(f"MSE: {mse:.4f}")
            # self.logger.info(f"MAPE(%): {mape:.4f}%")
            # self.logger.info(f"Accuracy (APE < 3%) for {origin_name}: {proportion_3}%")
            # self.logger.info(f"Accuracy (APE < 5%) for {origin_name}: {proportion_5}%")
            # self.logger.info(f"Accuracy (APE < 10%) for {origin_name}: {proportion_10}%")
            # self.logger.info(f"P90 APE(%) for {origin_name}: {p90_ape}%")
            # self.logger.info(f"P95 APE(%) for {origin_name}: {p95_ape}%")
            # self.logger.info(f"P99 APE(%) for {origin_name}: {p99_ape}%")
            # self.logger.info(f"MAX APE(%) for {origin_name}: {max_ape}%")
        self.logger.info(
            f"------------------------fold {self.configs['k_fold_order']}  performance details-------------------------------------------")
        table = tabulate(summary_data, headers=["metric"] + new_list, tablefmt='psql')
        self.logger.info(f"{table}")


    @calculate_running_time
    def run(self, train_with_all_data=False, label_scaler=None):
        self.logger.info("Begin to postprocess data")
        label_scaler = joblib.load(os.path.join(self.encoder_path, 'labels_minmax.pkl').replace("\\", "/"))


        x_train_raw_data = self.x_all_raw_data.loc[self.train_inds]
        x_test_raw_data = self.x_all_raw_data.loc[self.test_inds]
        y_train_raw_data =  self.label.loc[self.train_inds]
        y_test_raw_data = self.label.loc[self.test_inds]



        all_train_dataset = pd.concat([x_train_raw_data, y_train_raw_data], axis=1)


        true_scale_column = self.predicted_results.filter(like='True').columns.tolist()
        predict_scale_col =  self.predicted_results.filter(like='Predict').columns.tolist()
        if self.if_label_scale:
            true_values = self.predicted_results[true_scale_column].values
            predict_values = self.predicted_results[predict_scale_col].values
            self.predicted_results[true_scale_column] = pd.DataFrame(label_scaler.inverse_transform(true_values), columns=true_scale_column)
            self.predicted_results[predict_scale_col] = pd.DataFrame(label_scaler.inverse_transform(predict_values), columns=predict_scale_col)
        else:
            self.predicted_results[true_scale_column] = self.predicted_results[true_scale_column] * self.label_scale
            self.predicted_results[predict_scale_col] = self.predicted_results[predict_scale_col] * self.label_scale

        all_validate_dataset = pd.concat([x_test_raw_data.reset_index(drop=True), self.predicted_results], axis=1)

        if not self.train_with_all_data:
            processed_data = pd.concat([item for data_list in self.processed_data for item in data_list], axis=1)
            self.processed_data = processed_data
        x_proc_test_data = self.processed_data.loc[self.test_inds]
        all_proc_validate_dataset = pd.concat([x_proc_test_data.reset_index(drop=True), self.predicted_results], axis=1)
        self.save_data(all_proc_validate_dataset, "test_data_processed.csv")

        # convert nlc
        all_validate_dataset['RESULT.WorkloadName'] = all_validate_dataset['RESULT.WorkloadName'].str.replace('Memory Latency Checker latency', 'Memory Latency Checker')
        all_validate_dataset['RESULT.WorkloadName'] = all_validate_dataset['RESULT.WorkloadName'].str.replace(
            'Memory Latency Checker Bandwidth', 'Memory Latency Checker')

        all_train_dataset['RESULT.WorkloadName'] = all_train_dataset['RESULT.WorkloadName'].str.replace(
            'Memory Latency Checker latency', 'Memory Latency Checker')
        all_train_dataset['RESULT.WorkloadName'] = all_train_dataset['RESULT.WorkloadName'].str.replace(
            'Memory Latency Checker Bandwidth', 'Memory Latency Checker')

        # self.predicted_results['RESULT.WorkloadName'] = self.predicted_results['RESULT.WorkloadName'].str.replace(
        #     'Memory Latency Checker latency', 'Memory Latency Checker')
        # self.predicted_results['RESULT.WorkloadName'] = self.predicted_results['RESULT.WorkloadName'].str.replace(
        #     'Memory Latency Checker Bandwidth', 'Memory Latency Checker')


        self.overall_metric(self.true_col, self.predict_col)

        if self.configs["save_train_data"]:
            self.save_data(all_train_dataset, "train_data.csv")
        if self.configs["save_test_data"] and not train_with_all_data:
            self.save_data(all_validate_dataset, "test_data.csv")

            large_variation_data = all_validate_dataset.loc[all_validate_dataset.loc[:, all_validate_dataset.columns.str.contains('APE')].gt(5).any(axis=1)]

            self.save_data(large_variation_data, "large_variation_data.csv")
        # configs_name = "configs.yaml"
        # configs_save_path = os.path.join(self.config_save_path, configs_name).replace("\\", "/")
        # with open(configs_save_path, 'w') as file:
        #     yaml.dump(self.configs, file)
        # self.logger.info("Save configs to: {}".format(configs_save_path))
        if self.hist_all_list is not None:
            self.plot_validate_curve(hist_all_list=self.hist_all_list,path=self.output_path)


        return all_train_dataset, all_validate_dataset



    def plot_validate_curve(self, hist_all_list, path, metric='loss'):
        import matplotlib.pyplot as plt

        try:
            std = pd.concat(hist_all_list).groupby(level=0).apply(lambda x: x.std())
            average = sum(hist_all_list) / len(hist_all_list)
            save_path = os.path.join(path, "validate_curve.png").replace("\\", "/")
            epochs = range(1, std.shape[0] + 1)

            train_scores_mean = average[metric].to_list()
            train_scores_std = std[metric].to_list()
            test_scores_mean = average['val_' + metric].to_list()
            test_scores_std = std['val_' + metric].to_list()

            fig = plt.figure(figsize=(8, 6))
            ax = plt.axes()
            ax.set_title('Training and validation ' + metric)
            ax.set_ylim([0, 7000])
            ax.set_xlabel('Epoch', fontsize='large')
            ax.set_ylabel(metric, fontsize='large')

            ax.set_facecolor("#f2f2f2")
            ax.set_alpha(0.7)
            # img = plt.imread("C:/Users/Shino/OneDrive/xiaoman/projects/new_pp/stunning-guacamole/project report/figures/model/spec_fp/attention.png")
            # ax.imshow(img, extent=[0, 1, 0, 1], alpha=0.5, zorder=-1)
            fig.patch.set_facecolor("#f2f2f2")
            fig.patch.set_alpha(0.7)


            ax.grid(axis="both", linestyle='--', alpha=0.6, c='#d2c9eb', zorder=0)
            ax.fill_between(epochs, [x - y for x, y in zip(train_scores_mean, train_scores_std)],
                            [x + y for x, y in zip(train_scores_mean, train_scores_std)] , alpha=0.2,
                            color="darkorange", lw=2)
            ax.fill_between(epochs, [x - y for x, y in zip(test_scores_mean, test_scores_std)],
                            [x + y for x, y in zip(test_scores_mean, test_scores_std)], alpha=0.2,
                            color="navy", lw=2)

            ax.plot(epochs, train_scores_mean, '-', color="darkorange", label='Training ' + metric, lw=2)
            ax.plot(epochs, test_scores_mean, '-.', color="navy", label='Validation ' + metric, lw=2)

            ax.legend(['train', 'val'], loc='best')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logging.warning(f"Error plot metric: {e}")



