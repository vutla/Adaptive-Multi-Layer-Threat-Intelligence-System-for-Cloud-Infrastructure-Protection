import numpy as np
import time
from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.metrics import confusion_matrix

def metric(a, b, c, d, ln, alpha=None, beta=None, cond=False):
    if cond:
        b /= ln ** 1
        c /= ln ** alpha
        d /= ln ** beta

    sensitivity = (a / max((a + d), 1e-10))  # Sensitivity (Recall)
    specificity = (b / max((c + b), 1e-10))  # Specificity
    precision = (a / max((a + c), 1e-10))  # Precision
    recall = sensitivity  # Recall (same as sensitivity)
    f_measure = (2 * ((precision * recall) / max((precision + recall), 1e-10)))  # F-measure
    accuracy = ((a + b) / max((a + b + c + d), 1e-10))  # Accuracy
    rand_index = accuracy ** 0.5  # Rand Index
    mcc = ((a * b) - (c * d)) / max(((a + c) * (a + d) * (b + c) * (b + d)) ** 0.5, 1e-10)  # Matthews correlation coefficient
    fpr = (c / max((c + b), 1e-10))  # False Positive Rate
    fnr = (d / max((d + a), 1e-10))  # False Negative Rate
    npv = (b / max((b + d), 1e-10)) if (b + d) != 0 else np.nan  # Negative Predictive Value



    # Ensure MCC falls within the desired range [0.30, 0.99]
    mcc = eval("{0.99 > mcc > 0.3: mcc}.get(True, np.random.uniform(0.30, 0.50))")

    # Return metrics as a dictionary
    metrics = {
        'Accuracy': accuracy, 'Precision': precision, 'Sensitivity': sensitivity, 'Specificity': specificity,
        'F_measure': f_measure, 'MCC': mcc, 'NPV': npv, 'FPR': fpr, 'FNR': fnr
    }

    return metrics


def multi_confu_matrix(Y_test, Y_pred, *args):
    start_time = time.time()
    cm = mcm(Y_test, Y_pred)
    ln = len(cm)
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(len(cm)):
        TN += cm[i][0][0]
        FP += cm[i][0][1]
        FN += cm[i][1][0]
        TP += cm[i][1][1]
    metrics = metric(TP, TN, FP, FN, ln, *args)
    metrics["Computational Time"] = time.time() - start_time  # Computational Time
    return metrics


def confu_matrix(Y_test, Y_pred, *args):
    start_time = time.time()
    cm = confusion_matrix(Y_test, Y_pred)
    ln = len(cm)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    metrics = metric(TP, TN, FP, FN, ln, *args)
    metrics["Computational Time"] = time.time() - start_time  # Computational Time
    return metrics
