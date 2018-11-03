import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score


def get_roc_curve(true, score):
    fpr, tpr, _ = roc_curve(y_true=true, y_score=score)
    return fpr, tpr


def get_auc(fpr, tpr):
    return auc(fpr, tpr)


# gets roc_auc and plots if plotit=True
def get_roc(true, score, plotit=True):
    fpr, tpr = get_roc_curve(true, score)
    roc_auc = get_auc(fpr, tpr)
    if plotit:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc


def get_PRD_expansion(orig_df, id_name='LOAN_ID',
                      orig_name='ORIG_DTE', prd_name='PRD'):
    '''create full length dataframe with period dates in order to add macros'''
    '''ORIG_DTE: start date, PRD: last recorded date of loan'''
    # only get unique loans
    df_dates = orig_df[[id_name, orig_name, prd_name]].set_index(id_name)
    min_date, max_date = df_dates[orig_name].min(), df_dates[prd_name].max()
    keys = pd.date_range(min_date, max_date, freq='MS').values
    values = np.arange(len(keys))
    date2ind = dict(zip(keys, values))
    ind2date = dict(zip(values, keys))

    loan_ids_l, prd_l = [], []
    for loan_id in df_dates.index:
        vals = df_dates.loc[loan_id].values
        if isinstance(vals[0], np.ndarray):
            size = vals.shape[0]
            start, end = vals[0]
        else:
            size = 1
            start, end = vals
        inds_to_append = list(range(date2ind[start], date2ind[end]+1))
        prd_l += inds_to_append * size
        loan_ids_l += ([loan_id] * len(inds_to_append)) * size

    temp = pd.DataFrame({
        id_name: loan_ids_l,
        'PRD_I': prd_l}, dtype=np.int64).set_index(id_name)
    temp[prd_name] = temp['PRD_I'].apply(lambda x: ind2date[x])
    del temp['PRD_I']

    # with PRD in var_names PRD_x is current PRD, PRD_y is last PRD for loan
    temp = temp.merge(
        orig_df.drop_duplicates(id_name).set_index(id_name),
        how='left',
        left_index=True,
        right_index=True).reset_index()

    # rename did_dflt so only applies to last row of each loan
    temp['is_later'] = 1*(temp['PRD_x'] == temp['PRD_y'])
    temp['did_dflt'] = temp['is_later'] * temp['did_dflt']

    # remove extra PRD
    del temp['PRD_y'], temp['is_later'],
    return temp.rename(index=str, columns={'PRD_x': 'PRD'})


# check NAs and drop those with more than 1%
def drop_NA_cols(df, cutoff=0.01, excluded=[]):
    # check NAs
    col_na = df.apply(lambda x: x.isna().mean(), axis=0)
    drop_cols = col_na[col_na > cutoff]
    print('Drop candidates: \n{0}'.format(drop_cols))
    print('Excluded: {0}'.format(', '.join(excluded)))
    dropped = [v for v in drop_cols.index if v not in excluded]
    print('Dropped: {0}'.format(', '.join(dropped)))
    return df.drop(dropped, axis=1)

def split_train_test(df, test_ratio, key, p=None):
    """splits df by key into train and test set whose size is determined by test_ratio
    p is one-hot vector specifying which rows can be selected as test
    if key not provided, use index of df

    Args:
    df (pandas dataframe): dataframe
    test_ratio (float): % that will be test set
    key (str): key used to split test and train

    Returns:
    type: (test, train, p)
    """
    if key is None:
        # index is default index
        if p is None:
            p = np.ones(df.shape[0])
        test_size = int(df.shape[0] * test_ratio)
        if test_size == 0:
            print('ValueError: Invalid test size of 0. Reduce test_ratio or provide bigger dataset')
            sys.exit(1)
        test = df.sample(n=test_size, weights=p)
        is_test = df.index.isin(test.index)
        train = df[~is_test]
    else:
        # index is key
        index = df[key].unique()
        if p is None:
            p = np.ones(len(index))
        test_size = int(len(index) * test_ratio)
        if test_size == 0:
            print('ValueError: Invalid test size of 0. Reduce test_ratio or provide bigger dataset')
            sys.exit(1)
        test_index = np.random.choice(index, test_size,
                                      replace=False, p=p/sum(p))
        print('Test set: {0}'.format(', '.join([str(v) for v in test_index])))
        is_test = np.array([v in test_index for v in index])
        train_index = [v for v in index if v not in test_index]
        test, train = df[df[key].isin(test_index)], df[df[key].isin(train_index)]
    p[is_test] = 0
    print('Test size: {0}, Train size: {1}'.format(test.shape[0],
                                                   train.shape[0]))
    return test, train, p


def train_test_splitter(df, test_ratio, key=None):
    """generator that produces train, test set pairs
    no test set is repeated

    Args:
    df (pandas dataframe): dataframe
    test_ratio (float): % that will be test set
    key (str): key used to split test and train

    Yields:
    type: (test, train, p)
    """

    # calculate maximum number of times test set can be generated
    num_iterations = int(1/test_ratio)
    print('\nProducing {0} test sets...'.format(num_iterations))
    count = 1
    while count <= num_iterations:
        print('\n' + '='*80)
        print('='*80)
        print('\nTrain/Test split # {0}'.format(count))
        test, train, p = split_train_test(df,
                                          test_ratio,
                                          key)
        yield train, test, count
        count += 1
