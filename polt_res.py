import matplotlib.pyplot as plt
import numpy as np
from save_load import *
def polt_res():
    models = ['SVM + GA', 'RBFNN + RF', 'DNN', 'CNN + TSODE', 'Proposed']
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score',
               'FPR', 'FNR', 'MCC', 'NPV']

    data_70 = load('data_70')
    data_80 = load('data_80')
    # Plot each metric
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    x = np.arange(len(models))
    bar_width = 0.35

    for i in range(len(metrics)):
        plt.figure(figsize=(5.5, 4))
        plt.bar(x - bar_width / 2, data_70[:, i], width=bar_width, label='70 %', color='royalblue')
        plt.bar(x + bar_width / 2, data_80[:, i], width=bar_width, label='80 %', color='mediumseagreen')
        plt.xticks(x, models, rotation=15)
        plt.ylabel(metrics[i])
        plt.title(f'{metrics[i]} Comparison')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()


def Response_Latency():
    # Model names
    models = ['SVM + GA', 'RBFNN + RF', 'DNN', 'CNN + TSODE', 'Proposed']
    # Response latency values in ms
    latency_70 = load('latency_70')
    latency_80 = load('latency_80')
    # Bar settings
    x = np.arange(len(models))
    bar_width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - bar_width / 2, latency_70, width=bar_width, label='70 % Training', color='royalblue')
    plt.bar(x + bar_width / 2, latency_80, width=bar_width, label='80 % Training', color='mediumseagreen')
    # Plot styling
    plt.xticks(x, models, rotation=15)
    plt.ylabel("Response Latency (ms)")
    plt.title("Response Latency Comparison")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
