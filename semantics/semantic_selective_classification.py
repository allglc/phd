import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils import ECE_calc

def semantic_binning(df_calib, df_test, subdomain_attributes, class_names, min_count=3):

    print(subdomain_attributes)
    
    # learn subdomain perfo on calib set
    df_subdomains = df_calib.groupby(subdomain_attributes).agg(count=('correct', 'size'), accuracy=('correct', 'mean')).reset_index()
    print(f'{len(df_subdomains[df_subdomains["count"] >= min_count])} calib subdomains')

    # assign value to test subdomain according learned values
    df_test.loc[:, 'subdomain_score'] = None
    test_subdomains = df_test[subdomain_attributes].apply(tuple, 1)
    for subdomain, accuracy, count in zip(df_subdomains[subdomain_attributes].apply(tuple, 1), df_subdomains['accuracy'], df_subdomains['count']):
        if count >= min_count:
            idx = (test_subdomains == subdomain)
            df_test.loc[idx, 'subdomain_score'] = accuracy

    # for test subdomains not seen during calib, use average accuracy of the prediction
    accuracy_for_prediction = {k: 0 for k in class_names}
    accuracy_for_prediction.update(df_calib.groupby('prediction').agg(accuracy=('correct', 'mean'))['accuracy'].to_dict())
    idx_na = df_test['subdomain_score'].isna()
    print(f"{idx_na.sum()} nan values (they appear because subdomains in test are not all included in train). Fallback to average for predicted class.")
    df_test.loc[idx_na, 'subdomain_score'] = df_test['prediction'].replace(accuracy_for_prediction)

    print(
        f"AUROC baseline: {roc_auc_score(df_test['correct'].astype(float), df_test['confidence']):.3f}",
        f"AUROC subdomains: {roc_auc_score(df_test['correct'].astype(float), df_test['subdomain_score']):.3f}")
    print(
        f"ECE baseline: {ECE_calc(torch.tensor(np.stack((df_test['confidence'].to_numpy(), df_test['correct'].astype(float).to_numpy()), axis=1))).item():.3f}",
        f"ECE: {ECE_calc(torch.tensor(np.stack((df_test['subdomain_score'].astype(float).to_numpy(), df_test['correct'].astype(float).to_numpy()), axis=1))).item():.3f}")
    print()
    return df_test