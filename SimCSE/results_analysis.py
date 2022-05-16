# coding: utf-8
from collections import defaultdict
import pickle
import json
import math
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
try:
    import openpyxl
except ImportError:
    raise ImportError('You should install openpyxl')


markers = list(MarkerStyle.markers.keys())


def pivot_table(json_data_path: str, excel_save_path: str):
    data = json.load(open(json_data_path))
    df = pd.DataFrame(data)
    pivot_table = pd.pivot_table(df, index=list(df.columns.values)[:-1])
    pivot_table.to_excel(excel_save_path)


def plot(json_data_path: str, fig_save_path: str):
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)

    data = json.load(open(json_data_path))
    df = pd.DataFrame(data)

    group_models = list(df.groupby('model'))
    data2 = {}  # 相同数据集，相同sample方式，相同pooler type，不同模型之间的比较
    for item in group_models:
        model_name, group_model = item
        group_datasets = list(group_model.groupby('dataset'))
        for item in group_datasets:
            dataset_name, group_dataset = item
            data2.setdefault(dataset_name, {})
            group_pooler_types = list(group_dataset.groupby('pooler_type'))
            n_row = math.ceil(len(group_pooler_types) / 2)
            n_col = 2
            _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
            data1 = defaultdict(dict)  # 同一模型，相同数据集，相同sample，不同pooler type之间的对比
            for idx, item in enumerate(group_pooler_types):
                pooler_type, group_pooler_type = item
                x = idx // n_col
                y = idx % n_col
                if n_row > 1:
                    ax = axs[x, y]
                else:
                    ax = axs[y]
                group_samples = list(group_pooler_type.groupby('sample'))
                legends = []
                for idx, item in enumerate(group_samples):
                    sample, group_sample = item
                    legends.append(sample)
                    group_sample = group_sample.sort_values('dimension')
                    group_sample['dimension'] = group_sample['dimension'].apply(str)
                    data2[dataset_name].setdefault(sample, {})
                    data2[dataset_name][sample].setdefault(pooler_type, {})
                    data2[dataset_name][sample][pooler_type][model_name] = [group_sample['dimension'], group_sample['spearman']]
                    data1[sample][pooler_type] = [group_sample['dimension'], group_sample['spearman']]
                    marker = markers[idx % len(markers)]
                    # 同一模型，相同数据集，相同pooler type，不同sample之间的对比
                    ax.plot(group_sample['dimension'], group_sample['spearman'], marker=marker)
                ax.legend(legends)
                ax.set_title(pooler_type)
            fig_name = '{}_{}_compare_bettween_sample.png'.format(model_name, dataset_name)
            plt.savefig(os.path.join(fig_save_path, fig_name), dpi=300)
            plt.close()
            # 同一模型，相同数据集，相同sample，不同pooler type之间的对比
            n_row = math.ceil(len(data1) / 2)
            n_col = 2
            _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
            for idx, item in enumerate(data1.items()):
                sample, pooler_type2data = item
                x = idx // n_col
                y = idx % n_col
                if n_row > 1:
                    ax = axs[x, y]
                else:
                    ax = axs[y]
                legends = []
                for idx, item in enumerate(pooler_type2data.items()):
                    pooler_type, data = item
                    legends.append(pooler_type)
                    marker = markers[idx % len(markers)]
                    ax.plot(data[0], data[1], marker=marker)
                ax.legend(legends)
                ax.set_title(sample)
            fig_name = '{}_{}_compare_bettween_pooler_type.png'.format(model_name, dataset_name)
            plt.savefig(os.path.join(fig_save_path, fig_name), dpi=300)
            plt.close()
    # 相同数据集，相同sample方式，相同pooler type，不同模型之间的比较
    pkl_save_path = os.path.join(fig_save_path, 'compare_bettween_model.pkl')
    pickle.dump(data2, open(pkl_save_path, 'wb'), protocol=4)
    for dataset_name, sample2data in data2.items():
        for sample, pooler_type2data in sample2data.items():
            n_row = math.ceil(len(pooler_type2data) / 2)
            n_col = 2
            _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
            for idx, item in enumerate(pooler_type2data.items()):
                pooler_type, model_name2data = item
                x = idx // n_col
                y = idx % n_col
                if n_row > 1:
                    ax = axs[x, y]
                else:
                    ax = axs[y]
                legends = []
                for idx, item in enumerate(model_name2data.items()):
                    model_name, data = item
                    legends.append(model_name)
                    marker = markers[idx % len(markers)]
                    ax.plot(data[0], data[1], marker=marker)
                ax.legend(legends)
                ax.set_title(pooler_type)
            fig_name = '{}_{}_compare_bettween_model.png'.format(dataset_name, sample)
            plt.savefig(os.path.join(fig_save_path, fig_name), dpi=300)
            plt.close()


def test(json_data_path: str):
    data = json.load(open(json_data_path))
    df = pd.DataFrame(data)

    group_models = list(df.groupby('model'))
    data2 = {}  # 相同数据集，相同sample方式，相同pooler type，不同模型之间的比较
    for item in group_models:
        model_name, group_model = item
        group_datasets = list(group_model.groupby('dataset'))
        for item in group_datasets:
            dataset_name, group_dataset = item
            data2[dataset_name] = {}
            group_pooler_types = list(group_dataset.groupby('pooler_type'))
            for idx, item in enumerate(group_pooler_types):
                pooler_type, group_pooler_type = item
                group_samples = list(group_pooler_type.groupby('sample'))
                for idx, item in enumerate(group_samples):
                    sample, group_sample = item
                    group_sample = group_sample.sort_values('dimension')
                    group_sample['dimension'] = group_sample['dimension'].apply(str)
                    if sample not in data2[dataset_name]:
                        data2[dataset_name][sample] = {}
                    if pooler_type not in data2[dataset_name][sample]:
                        data2[dataset_name][sample][pooler_type] = {}
                    data2[dataset_name][sample][pooler_type][model_name] = [group_sample['dimension'], group_sample['spearman']]


if __name__ == '__main__':
    json_data_path = '/source/c0/NLPSource/embedding/simcse/simcse_results_with_dimension_reduction.json'
    # excel_save_path = '/source/c0/NLPSource/embedding/simcse/simcse_results_with_dimension_reduction.xlsx'
    # pivot_table(json_data_path, excel_save_path)
    fig_save_path = '/source/c0/NLPSource/embedding/simcse/simcse_results_with_dimension_reduction_figs'
    plot(json_data_path, fig_save_path)
    # test(json_data_path)
