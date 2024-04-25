import sys

import pandas as pd
import numpy as np
import os
import logging


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

    if configs["select_model"] == "LinearModel":
        merged_coef_data = pd.concat(merged_coef, axis=1)
        merged_coef_data = merged_coef_data.loc[:, ~merged_coef_data.columns.duplicated()]
        merged_coef_data.to_csv(coef_output_path, index=False)
    return variation_data, test_results

def results_evaluate(configs, test_results):
    results = pd.DataFrame()
    true_name_list = configs["true_col"]
    predict_name_list = configs["predict_col"]
    origin_name_list = configs["label_name_list"]
    model_name = configs["select_model"]
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
        logger.info(f"------------------------{origin_name}_over_all performance details-------------------------------------------")
    #
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAPE(%): {mape:.4f}%")
        logger.info(f"Accuracy (APE < 3%) for {origin_name}: {proportion_3}%")
        logger.info(f"Accuracy (APE < 5%) for {origin_name}: {proportion_5}%")
        logger.info(f"Accuracy (APE < 10%) for {origin_name}: {proportion_10}%")
        logger.info(f"P90 APE(%) for {origin_name}: {p90_ape}%")
        logger.info(f"P95 APE(%) for {origin_name}: {p95_ape}%")
        logger.info(f"P99 APE(%) for {origin_name}: {p99_ape}%")
        logger.info(f"MAX APE(%) for {origin_name}: {max_ape}%")



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
"""