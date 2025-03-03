from scipy.stats import bootstrap
from sklearn.metrics import confusion_matrix
import numpy as np

def calculate_macro_average(values):
    if isinstance(values[0], tuple):
        ci_lower, ci_upper = map(list, [*zip(*values)])
        return (sum(ci_lower)/len(ci_lower), sum(ci_upper)/len(ci_upper))
            
    return sum(values)/len(values)

def calculate_weighted_average(values, weights):
    if isinstance(values[0], tuple):
        ci_lower, ci_upper = map(list, [*zip(*values)])
        ci_lower_weighted_values = [value*weight for value, weight in zip(ci_lower, weights)]
        ci_upper_weighted_values = [value*weight for value, weight in zip(ci_upper, weights)]

        return (sum(ci_lower_weighted_values)/sum(weights), sum(ci_upper_weighted_values)/sum(weights))
    
    weighted_values = [value*weight for value, weight in zip(values, weights)]
    return sum(weighted_values)/sum(weights)

def get_confusion_matricies(y_true, y_pred):
    """
    Get Confusion Matrix for each class in a multi-class classification using 1vsRest Approach.

    Parameters:
    - y_true: Array-like of true labels
    - y_pred: Array-like of predicted labels

    Returns:
    A list of confusion matrices
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # print(f"\n\nConfusion Matrx: \n{cm}\n\n")
    num_classes = cm.shape[0]
    cm_list = list()

    if num_classes > 2:
        for i in range(num_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (FP + FN + TP)
            cm_list.append(np.array([[TN, FP],[FN, TP]]))
    else:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_list.append(cm)
    # print(f"cm_dict: {cm_list}")
    return cm_list

def calc_acc(conf, p=None):
    tn, fp, fn, tp = conf.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    return acc

def calc_se(conf, p=None):
    tn, fp, fn, tp = conf.ravel()
    se = tp / (tp + fn) if tp + fn != 0 else 0
    return se

def calc_sp(conf, p=None):
    tn, fp, fn, tp = conf.ravel()
    sp = tn / (tn + fp) if tn + fp != 0 else 0
    return sp

def calc_p(conf, p=None):
    tn, fp, fn, tp = conf.ravel()
    p = (tp + fn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    return p

def calc_npv(conf, p=None):
    # tn, fp, fn, tp = pred_label_pairs_to_conf(pairs).ravel()
    sp = calc_sp(conf)
    se = calc_se(conf)
    p = p if p is not None else calc_p(conf)
    npv = (sp*(1-p))/((1-se)*p+sp*(1-p)) if (1-se)*p+sp*(1-p) != 0 else 0
    return npv

def calc_ppv(conf, p=None):
    # tn, fp, fn, tp = conf_mat.ravel()
    sp = calc_sp(conf)
    se = calc_se(conf)
    p = p if p is not None else calc_p(conf)
    ppv = (se*p)/((se*p)+((1-sp)*(1-p))) if (se*p)+((1-sp)*(1-p)) != 0 else 0
    return ppv

def make_full_stat_func(stat_func, comb_func, weight_func=None):
    def f(true_pred_pairs, axis=None):
        print(true_pred_pairs)
        try:
            true, pred = zip(*true_pred_pairs)
        except TypeError as e:
            
            raise e
        mats = get_confusion_matricies(true, pred)
        stats = [stat_func(m) for m in mats]
        val = comb_func(stats)
        return val
    return f


def calc_all_stats_and_ci(pairs):
    params = zip(('acc', 'se', 'sp', 'p', 'npv', 'ppv'),
                (calc_acc, calc_se, calc_sp, calc_p, calc_npv, calc_ppv),)
    for (name, stat_func) in params:
        metric_calc = make_full_stat_func(stat_func, calculate_macro_average)
        # exact_val = metric_calc(pairs)
        data = (pairs,)
        print(data)
        result = bootstrap(data, metric_calc)

if __name__ == '__main__':
    # TODO: Return actual results
    predictions = []

    # FIXME: Generates fake results
    np.random.seed(243)
    count = 250
    label_idx_map = {'class0':0, 'class1':1, 'class2':2, }
    for i in range(count):
        filename = f'testimg_{i}'
        probs = np.random.rand(len(label_idx_map))
        probs = probs / probs.sum()
        y_hat = np.argmax(probs)
        if np.random.random() < 0.7:
            label = y_hat
        else:
            label = np.random.randint(0, len(label_idx_map))

        predictions.append({
            'id':filename,
            'probs':probs.tolist(),
            'y_hat':int(y_hat),
            'label':int(label),
        })
        
    result = {
        'map':label_idx_map,
        'predictions':predictions
    }

    pred_pairs = [(v['label'], v['y_hat']) for v in result['predictions']]
    calc_all_stats_and_ci(pred_pairs)