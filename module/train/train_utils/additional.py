#Copyright (C) <2024> Intel Corporation
#SPDX-License-Identifier: Apache-2.0


import sys

import pandas as pd
import numpy as np
import os
import logging
from tabulate import tabulate
import re




logger = logging.getLogger("Addition")

def merge_K_fold_results(configs):
    logger.info("begin merge data")
    path = configs["output_path"].replace("\\", "/")
    variation_output = 'summary_variation.csv'
    test_data_output = 'summary_test_data.csv'
    coef_output = 'summary_model_coef.csv'

    merged_variation = []
    merged_test_data = []
    merged_coef = []
    for i in range(1, configs["n_split"] + 1):
        variation_i_path = os.path.join(path, eval(configs["split_folder"]).replace("\\", "/"),
                                        'large_variation_data.csv')
        test_data_i_path = os.path.join(path, eval(configs["split_folder"]).replace("\\", "/"), 'test_data.csv')
        model_coef_i_path = os.path.join(path, eval(configs["split_folder"]).replace("\\", "/"), 'model_coef.csv')
        if os.path.exists(variation_i_path):
            variation_i_data = pd.read_csv(variation_i_path)
            variation_i_data["data_tag"] = i
            merged_variation.append(variation_i_data)
        if os.path.exists(test_data_i_path):
            test_i_data = pd.read_csv(test_data_i_path)
            test_i_data["data_tag"] = i
            merged_test_data.append(test_i_data)
        if configs["select_model"] == "LinearModel":
            if os.path.exists(model_coef_i_path):
                model_coef = pd.read_csv(model_coef_i_path)
                model_coef = model_coef.rename(columns={"Coefficient":f"Fold-{i}_coef"})
                merged_coef.append(model_coef)

    variation_data = pd.concat(merged_variation)
    test_results = pd.concat(merged_test_data)

    variation_output_path = os.path.join(path, variation_output)
    test_data_output_path = os.path.join(path, test_data_output)
    coef_output_path = os.path.join(path, coef_output).replace("\\", "/")

    variation_data.to_csv(variation_output_path, index=False)
    test_results.to_csv(test_data_output_path, index=False)
    logger.info(f"Merged {configs['n_split']} Fold results into path {path}")
    logger.info("--------------------Total metrics of K_fold is:-----------------------")
    results_evaluate(configs, test_results)

    if merged_coef:
        merged_coef_data = pd.concat(merged_coef, axis=1)
        merged_coef_data = merged_coef_data.loc[:, ~merged_coef_data.columns.duplicated()]
        merged_coef_data["workload_name"] = configs["workload_names"][0]
        merged_coef_data.to_csv(coef_output_path, index=False)
    return variation_data, test_results

def results_evaluate(configs, test_results):
    results = pd.DataFrame()
    true_name_list = configs["true_col"]
    predict_name_list = configs["predict_col"]
    origin_name_list = configs["label_name_list"]
    model_name = configs["select_model"]
    summary_data = pd.DataFrame(columns=["metric"] + origin_name_list)
    summary_data["metric"] = ["MAE", "MSE", "MAPE(%)",
                              "Accuracy (APE < 3%)", "Accuracy (APE < 5%)", "Accuracy (APE < 10%)",
                              "P90 APE(%)", "P95 APE(%)", "P99 APE(%)", "MAX APE(%)"]



    # Calculate the Absolute Error (AE) for dataframe
    for i in range(len(true_name_list)):
        label_name = true_name_list[i]
        predict_name = predict_name_list[i]
        origin_name = origin_name_list[i]

        test_results = test_results.reset_index(drop=True)
        ae = abs(test_results[label_name] - test_results[predict_name])
        # Calculate the Mean Absolute Error (MAE)
        se = (test_results[label_name] - test_results[predict_name]) ** 2

        # Calculate the Mean Absolute Percentage Error (MAPE)
        ape = abs(test_results[label_name] - test_results[predict_name]) / test_results[label_name] * 100

        result = pd.concat([test_results[label_name], test_results[predict_name], ae, se, ape], axis=1)
        result.columns = [f'{label_name}', f'{predict_name}', f'AE_{label_name}_{model_name}',
                          f'SE_{label_name}_{model_name}', f'APE(%)_{label_name}_{model_name}']
        results = pd.concat([results, result], axis=1)

    # Calculate the Absolute Error (AE)
    # y_test = y_test.reset_index(drop=True)
    # y_predict = y_predict.reset_index(drop=True)
    # ae = abs(y_test["True_value"] - y_predict["Predict_value"])
        mae = np.mean(ae)
    # # Calculate the Mean Absolute Error (MAE)
    # se = (y_test["True_value"] - y_predict["Predict_value"]) ** 2
        mse = np.mean(se)
    # # Calculate the Mean Absolute Percentage Error (MAPE)
    # ape = abs(y_test["True_value"] - y_predict["Predict_value"]) / y_test["True_value"] * 100
        mape = np.mean(ape)
        p50_ape = round(ape.quantile(0.5), 4)
        p90_ape = round(ape.quantile(0.9), 4)
        p95_ape = round(ape.quantile(0.95), 4)
        p99_ape = round(ape.quantile(0.99), 4)
        max_ape = round(np.max(ape), 4)
        count_3 = (ape < 3).sum()
        proportion_3 = round(count_3 / len(ape) * 100, 5)
    #
        count_5 = (ape < 5).sum()
        proportion_5 = round(count_5 / len(ape) * 100, 4)
    #
        count_10 = (ape < 10).sum()

        proportion_10 = round(count_10 / len(ape) * 100, 4)
        summary_data[origin_name] = [mae, mse, mape,
                                     proportion_3, proportion_5, proportion_10,
                                     p90_ape, p95_ape, p99_ape, max_ape]

    #
    #     logger.info(f"------------------------{origin_name}_over_all performance details-------------------------------------------")
    # #
    #     logger.info(f"MAE: {mae:.4f}")
    #     logger.info(f"MSE: {mse:.4f}")
    #     logger.info(f"MAPE(%): {mape:.4f}%")
    #     logger.info(f"Accuracy (APE < 3%) for {origin_name}: {proportion_3}%")
    #     logger.info(f"Accuracy (APE < 5%) for {origin_name}: {proportion_5}%")
    #     logger.info(f"Accuracy (APE < 10%) for {origin_name}: {proportion_10}%")
    #     logger.info(f"P90 APE(%) for {origin_name}: {p90_ape}%")
    #     logger.info(f"P95 APE(%) for {origin_name}: {p95_ape}%")
    #     logger.info(f"P99 APE(%) for {origin_name}: {p99_ape}%")
    #     logger.info(f"MAX APE(%) for {origin_name}: {max_ape}%")
    logger.info(
        f"------------------------total {configs['n_split']} Fold summary performance details-------------------------------------------")
    table = tabulate(summary_data, headers=["metric"] + origin_name_list, tablefmt='psql')
    logger.info(f" \n {table}")
def convert_data(data):

    test_name_columns = data.columns.tolist()
    desired_columns = [col for col in test_name_columns if 'True' in col or 'Predict' in col or 'AE' in col or 'SE' in col or 'APE' in col]
    static_cols = list(set(test_name_columns) - set(desired_columns))

    melted_df = data.melt(id_vars=static_cols,
                          value_vars=desired_columns,
                          var_name='RESULT.TestName', value_name='Value')
    melted_df = melted_df.loc[:, ~melted_df.columns.duplicated()]
    melted_df['RESULT.TestName'] = melted_df['RESULT.TestName'].astype(str)
    # Extract the relevant information from the TEST_NAME column
    melted_df['TEST_TYPE'] = melted_df['RESULT.TestName'].str.extract('(True|Predict|AE|SE|APE)', expand=False)
    melted_df['RESULT.TestName'] = melted_df['RESULT.TestName'].apply(lambda x: x.split('_', 1)[1], )
    # melted_df['TEST_NAME'] = melted_df['TEST_NAME'].str.extract('(.*)\s+(true|predict|AE|SE|ape)', expand=False)[1]
    # + ["RESULT.TestName"]
    data = melted_df.pivot_table(index=static_cols,
                                 columns=['TEST_TYPE'], values='Value', aggfunc='first').reset_index()

    return data

def param_search(x_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBRegressor
    model = XGBRegressor()

    # Define the parameters to search for
    # param_grid = {
    #     "n_estimators": [100, 200, 300, 400, 500],
    #     "max_depth": [10, 20, 30, 40, 50],
    #     "learning_rate": [0.01, 0.05, 0.1, 0.2],
    #     "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
    #     "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    #     "reg_lambda": [0.1, 0.5, 1.0, 1.5, 2.0],
    #     "eta": [0.3, 0.2, 0.1]
    # }

    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [ 30, 40, 50],
        "learning_rate": [ 0.1, 0.2],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
        "reg_lambda": [0.1, 0.5, 1.0, 1.5, 2.0],
        "eta": [0.3, 0.2, 0.1]
    }

    # Conduct the grid search with 5-fold cross-validation
    logger.info("Begin param search")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    logger.debug(f"Begin param search, param grid is {param_grid}")
    grid_search.fit(x_train, y_train)

    logger.debug("Best parameters: ", grid_search.best_params_)
    logger.debug("Best score: ", grid_search.best_score_)
    sys.exit(1)

def multi_label_transfer(raw_data, unfold_col_name, unfold_value_col_name):
    column_names = raw_data.columns[(raw_data.columns != unfold_col_name) & (raw_data.columns != unfold_value_col_name)].tolist()

    df_pivot = raw_data.pivot(index=column_names, columns=unfold_col_name,values=unfold_value_col_name).reset_index()
    df_pivot = df_pivot.reset_index(drop=True)
    df_pivot = df_pivot.dropna().reset_index(drop=True)
    # df_pivot_deduped = df_pivot.drop_duplicates(subset=column_names)
    unfold_col_name_values = raw_data[unfold_col_name].unique()
    # Convert the resulting array to a list
    unfold_col_name_list = unfold_col_name_values.tolist()
    # df_pivot.columns = column_names + unfold_col_name_list
    return df_pivot, unfold_col_name_list

def linpack_transfer(raw_data, unfold_col_name, unfold_value_col_name):
    column_names = raw_data.columns[(raw_data.columns != unfold_col_name) & (raw_data.columns != unfold_value_col_name)].tolist()
    df_pivot = raw_data.pivot(index=column_names, columns=unfold_col_name,values=unfold_value_col_name).reset_index()
    df_pivot = df_pivot.dropna().reset_index(drop=True)
    # df_pivot_deduped = df_pivot.drop_duplicates(subset=column_names)
    unfold_col_name_values = raw_data[unfold_col_name].unique()
    # Convert the resulting array to a list
    unfold_col_name_list = ["Score(GFLOPS)"]
    # df_pivot.columns = column_names + unfold_col_name_list
    return df_pivot, unfold_col_name_list

def MLC_multi_label_transfer(raw_data, unfold_col_name, unfold_value_col_name):
    column_names = raw_data.columns[
        (raw_data.columns != unfold_col_name) & (raw_data.columns != unfold_value_col_name) & (raw_data.columns != "RESULT.WorkloadPreset")
         & (raw_data.columns != "kubernetes.pod_id")  & (raw_data.columns != "SVR.Memory.MemFree")].tolist()
    raw_data = raw_data.drop(["RESULT.WorkloadPreset"], axis=1)
    raw_data = raw_data.drop(["kubernetes.pod_id"], axis=1)
    raw_data = raw_data.drop(["SVR.Memory.MemFree"], axis=1)
    feature_num = len(column_names)
    label_num = raw_data[unfold_col_name].unique().tolist()
    # column_names = ["Measure.DIMM.PartNo", "META.metadata.cscope.qdf0", "SVR.Memory.MemFree", "META.metadata.cscope.stepping", "Measure.DIMM.Population", "Measure.DIMM.Total", "Measure.DIMM.Num", "SVR.CPU.Prefetchers", "RESULT.IterationIndex", "RESULT.kubernetes.host"]
    df_pivot = raw_data.pivot(index=column_names, columns=unfold_col_name,values=unfold_value_col_name).reset_index()
    df_pivot = df_pivot.dropna().reset_index(drop=True)

    group_df = df_pivot.groupby(column_names).sum()
    index_df = group_df.index.to_frame().reset_index(drop=True)
    merge_all = pd.concat([index_df, group_df.reset_index(drop=True)], axis=1)

    unfold_col_name_values = raw_data[unfold_col_name].unique()
    # Convert the resulting array to a list
    unfold_col_name_list = unfold_col_name_values.tolist()
    # df_pivot.columns = column_names + unfold_col_name_list
    return merge_all, unfold_col_name_list

def read_current_dimm_part_num():
    excel_file_path = "C:/Users/xiaomanl/OneDrive - Intel Corporation/Documents/project/stunning-guacamole/data/external/DIMM vendor_pn_rank info.xlsx"
    sheet_name = 'DIMM vendor_pn_rank info'
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    # new_features  = pd.DataFrame(df["DIMM.PartNumber"].apply(self.decode_dimm_part_number).tolist())



    df.to_csv("C:/Users/xiaomanl/OneDrive - Intel Corporation/Documents/project/stunning-guacamole/data/external/DIMM_process.csv")




def plot_permutation_importance(clf, X, y, ax):
    from sklearn.inspection import permutation_importance
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax

def impurity_feature_importance(clf, X_train, y_train):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    mdi_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
    tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    mdi_importances.sort_values().plot.barh(ax=ax1)
    ax1.set_xlabel("Gini importance")
    plot_permutation_importance(clf, X_train, y_train, ax2)
    ax2.set_xlabel("Decrease in accuracy score")
    fig.suptitle(
        "Impurity-based vs. permutation importances on multicollinear features (train set)"
    )
    _ = fig.tight_layout()

def permutation_importance(clf, X_test,  y_test):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_permutation_importance(clf, X_test, y_test, ax)
    ax.set_title("Permutation Importances on multicollinear features\n(test set)")
    ax.set_xlabel("Decrease in accuracy score")
    _ = ax.figure.tight_layout()

def headmap_feature(raw_features, save_path):
    import matplotlib.pyplot as plt
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr
    cols = raw_features.select_dtypes(include=['int', 'float']).columns.to_list()
    features = raw_features[cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    constant_columns = [col for col in features.columns if features[col].nunique() == 1]

    df = features.drop(columns=constant_columns)

    corr = spearmanr(df).correlation
    s_corr = features.corr(method='spearman')

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    # np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    # distance_matrix_df = pd.DataFrame(distance_matrix)
    np.fill_diagonal(distance_matrix, 0)


    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=df.columns.to_list(), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    # ax2.imshow(corr.iloc[dendro["leaves"], :][:, dendro["leaves"]])
    corr_df = pd.DataFrame(corr, columns=df.columns, index=df.columns)
    ax2.imshow(corr_df.iloc[dendro["leaves"], :].iloc[:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path)




def group_features2cluster():
    from collections import defaultdict


    cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = X.columns[selected_features]

    X_train_sel = X_train[selected_features_names]
    X_test_sel = X_test[selected_features_names]

    clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_sel.fit(X_train_sel, y_train)
    print(
        "Baseline accuracy on test data with features removed:"
        f" {clf_sel.score(X_test_sel, y_test):.2}"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_permutation_importance(clf_sel, X_test_sel, y_test, ax)
    ax.set_title("Permutation Importances on selected subset of features\n(test set)")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    plt.show()


"""
conda install -c anaconda pandas -y
conda install -c conda-forge pyyaml -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda joblib -y
conda install -c conda-forge tensorflow -y
conda install -c anaconda pymysql -y
conda install -c anaconda scikit-learn -y
conda install -c anaconda statsmodels -y
conda install -c conda-forge tabulate -y
conda install -c conda-forge xgboost -y
conda install -c anaconda openpyxl -y
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow-gpu==2.10.0
pennylane==0.23
python -m pip install PennyLane==0.23.1
conda install -c nvidia cuda-nvcc
"""