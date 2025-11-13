from sklearn import metrics
import numpy as np


def get_cm_values(y_true, y_pred, num_classes):
    MCM = metrics.multilabel_confusion_matrix(y_true, y_pred, labels=range(num_classes))
    tn = MCM[:, 0, 0]
    fn = MCM[:, 1, 0]
    tp = MCM[:, 1, 1]
    fp = MCM[:, 0, 1]
    return tn, fn, tp, fp

def _prf_divide(numerator, denominator, zero_division=0):
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1 # avoid infs/nans
    result = numerator / denominator
    if not np.any(mask):
        return result
    result[mask] = 0.0 if zero_division in [0] else 1.0
    return result

def compute_fairness_metrics(lighter_labels, lighter_predictions, darker_labels, darker_predictions, num_classes, zero_division=0):
    tn, fn, tp, fp = get_cm_values(lighter_labels, lighter_predictions, num_classes)
    lighter_TPR = _prf_divide(tp, tp + fn, zero_division)
    lighter_TNR = _prf_divide(tn, tn + fp, zero_division)
    lighter_FPR = _prf_divide(fp, tn + fp, zero_division) 
        
    tn, fn, tp, fp = get_cm_values(darker_labels, darker_predictions, num_classes)
    darker_TPR = _prf_divide(tp, tp + fn, zero_division)
    darker_TNR = _prf_divide(tn, tn + fp, zero_division)
    darker_FPR = _prf_divide(fp, tn + fp, zero_division) 
        
    # equalized opportunity
    eopp0 = np.average(np.abs(lighter_TNR - darker_TNR))
    eopp1 = np.average(np.abs(lighter_TPR - darker_TPR))

    # equalized odds
    eodd = np.abs(((lighter_TPR - darker_TPR) + (lighter_FPR - darker_FPR))).mean() / 2
    return eopp0, eopp1, eodd


def calculate_fairness_metrics(predictions, labels, fitz_types, num_types):
    labels_array = np.zeros((num_types, 6)) # len(set(labels))
    correct_array = np.zeros((num_types, 6))
    predictions_array = np.zeros((num_types, 6))  
    
    # 원본코드     
    for pred, label, fitz in zip(predictions, labels, fitz_types):
        if num_types == 2 and (fitz == 3.0 or fitz == 4.0):
            type_index = int(fitz) - 3
        else:
            type_index = int(fitz) - 1  
    
    # 20241230 type index Fitzpatrick Scale 값이 예상 범위를 벗어나는 경우를 처리 (grad_cam_plus 적용용)
    # for pred, label, fitz in zip(predictions, labels, fitz_types):
    #     if num_types == 2 and (fitz == 3.0 or fitz == 4.0):
    #         type_index = int(fitz) - 3  # 3, 4는 0, 1로 매핑
    #     elif 1 <= fitz <= num_types:
    #         type_index = int(fitz) - 1  # 1~4을 0~3로 매핑
    #     else:
    #         raise ValueError(f"Unexpected fitz value: {fitz}")

        labels_array[type_index, label] += 1
        predictions_array[type_index, pred] += 1
        if pred == label:
            correct_array[type_index, label] += 1
            
    correct_sum_per_type = np.sum(correct_array, axis=1)
    labels_sum_per_type = np.sum(labels_array, axis=1)
    acc_per_type = correct_sum_per_type / labels_sum_per_type
    avg_acc = np.sum(correct_array) / np.sum(labels_array)

    # PQD 
    PQD = acc_per_type.min() / acc_per_type.max()

    # DPM 
    demo_array = predictions_array / np.sum(predictions_array, axis=1, keepdims=True)
    demo_max = np.where(demo_array.max(axis=0)==0, 1, demo_array.max(axis=0))
    DPM = np.mean(demo_array.min(axis=0) / demo_max)

    # EOM 
    with np.errstate(divide='ignore', invalid='ignore'):
        eo_array = correct_array / labels_array
        eo_array[~np.isfinite(eo_array)] = 0
    eo_max = np.where(eo_array.max(axis=0)==0, 1, eo_array.max(axis=0))
    EOM = np.mean(np.min(eo_array, axis=0) / eo_max)

    return {'acc_avg': avg_acc, 'acc_per_type': acc_per_type, 'PQD': PQD, 'DPM': DPM, 'EOM': EOM}
