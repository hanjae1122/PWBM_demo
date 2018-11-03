import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import resample

def _btstrp_data(data, key='ORIG_DTE'):
    ############################################################
    # Should preserve entire range for each vintage? Currently, no
    ############################################################
    return resample(data)


def get_btstrp(pred_fxn, model_specs, train, test,
               btstrp_trials=30, lo=2.5):
    '''returns df (test.shape[0], 2) that is '''
    '''the lo, hi percentile btstrp prediction'''
    '''pred_fxn should return test set prediction'''
    ############################################################
    # FIX: BOOTSTRP samples must preserve entire period of loan
    ############################################################
    hi = 100 - lo
    # initial df
    btstrp_all = pd.DataFrame()
    for btstrp_trial_i in range(btstrp_trials):
        print('\nBootstrp trial {0}'.format(btstrp_trial_i))
        btstrp_train = _btstrp_data(train)
        _, preds = pred_fxn(model_specs, btstrp_train, test)
        to_concat = pd.DataFrame({'btstrp_pred': preds}, index=test.index)
        if btstrp_all.shape[0] == 0:
            btstrp_all = to_concat
        else:
            btstrp_all = pd.concat([btstrp_all, to_concat], axis=1)

    print('\nCompiling bootstrap data')
    # get lo, hi percentiles by row and split the tuple into 2 column df
    btstrp_rslt = btstrp_all.apply(lambda x: (np.percentile(x, lo),
                                              np.percentile(x, hi)),
                                   axis=1).apply(pd.Series)
    btstrp_rslt.columns = ['{0}%'.format(lo), '{0}%'.format(hi)]
    return btstrp_rslt


def plot_btstrp(for_plt, x, y, pred_y, lo_name, hi_name, ax=None):
    print('Generating plot')
    if ax is None:
        ax = plt.subplot()
    sns.lineplot(x=x, y=lo_name, data=for_plt, linewidth=0.5,
                 ax=ax, color='#75bbfd', linestyle='--')
    sns.lineplot(x=x, y=hi_name, data=for_plt, linewidth=0.5,
                 ax=ax, color='#75bbfd', linestyle='--', label='CI')
    l1, l2 = ax.lines[0], ax.lines[1]

    # Get the xy data from the lines so that we can shade
    d1, d2 = l1.get_xydata(), l2.get_xydata()
    ax.fill_between(d1[:, 0], y1=d1[:, 1], y2=d2[:, 1],
                    color="#95d0fc", alpha=0.3)
    sns.lineplot(x=x, y=pred_y, data=for_plt,
                 ax=ax, linewidth=1.5, label='Prediction', color='#0165fc')
    sns.lineplot(x=x, y=y, data=for_plt,
                 ax=ax, linewidth=1.5, label='Actual', color='#c04e01')
    # ax.set_title('Cumulative Prediction [{0}, {1}]'.format(lo_name, hi_name))
