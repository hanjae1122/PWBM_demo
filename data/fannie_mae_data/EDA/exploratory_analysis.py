# import packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)

#%%
# define global functions
def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    valid = s[~s.isna()]
    dates = {date: pd.to_datetime(date,
                                  infer_datetime_format=True).to_period('m')
             for date in valid.unique()}
    dates[np.nan] = np.nan
    return s.map(dates)

#%%
# set PATH variables
PATH = os.path.join(os.getcwd(), 'Documents', 'GitHub', 'PWBM_demo')
CURR_PATH = os.path.join(PATH, 'data', 'fannie_mae_data', 'EDA')
CLEAN_PATH = os.path.join(PATH, 'data', 'fannie_mae_data', 'clean')
EXPORT_PATH = os.path.join(CURR_PATH, 'results')

#%%
# Reading data
with open(os.path.join(CURR_PATH, 'EDA_filelist.txt')) as f:
        filelist, yearlist = [], []
        for line in f:
            year_files = line.split()
            try:
                yearlist.append(int(year_files[0]))
            except ValueError:
                print('ValueError: First value in each'
                      'row of filelist must be a year')
                sys.exit(1)
            filelist.append(year_files[1:])

with open(os.path.join(CURR_PATH, 'EDA_varlist.txt')) as f:
    varlist = {'CAT': [], 'CONT': []}
    for line in f:
        type_vars = line.split()
        typekey = type_vars[0].upper().replace(' ', '')
        try:
            varlist[typekey] = type_vars[1:]
        except KeyError:
            print('KeyError: Each row of varlist must'
                  'start with the keyword CAT or CONT')
            sys.exit(1)
            
#%%
cat_vars, cont_vars = varlist['CAT'], varlist['CONT']
del varlist

print('Reading files: {0}'.format([', '.join(l) for l in filelist]))
print('Continuous variables: '
      '{0}\nCategorical variables: {1}'.format(', '.join(cont_vars),
                                               ', '.join(cat_vars)))

#%%
# read filelist one batch at a time and aggregate to df
# remove rows not from year
df = pd.DataFrame()
for i in range(len(filelist)):
        filebatch = filelist[i]
        print('\n' + '=' * 80)
        print('\nProcessing current batch...: {0}'.format(filebatch))
        batch_to_concat = pd.DataFrame()
        for filename in filebatch:
            print('\nReading {0}...'.format(filename))
            # read monthly loan data
            to_concat = pd.read_csv(os.path.join(CLEAN_PATH, filename),
                                    engine='c')

            batch_to_concat = pd.concat([batch_to_concat, to_concat], axis=0)
        print('Total number of rows in batch: {0}'
              .format(batch_to_concat.shape[0]))

        batch_to_concat['ORIG_DTE'] = lookup(batch_to_concat['ORIG_DTE'])

        year = yearlist[i]
        print('\nRemoving rows not in year {0}...'.format(year))

        # remove vintages not from the year in list    
        batch_to_concat['yORIG_DTE'] = (batch_to_concat['ORIG_DTE']
                                        .apply(lambda x: int(x.year)))
        a = batch_to_concat.shape
        batch_to_concat = batch_to_concat[batch_to_concat['yORIG_DTE'] == year]
        print('With year restriction, retained {0:.2f}% of {1} loans'
              .format(100 * batch_to_concat.shape[0]/a[0], a[0]))
        del batch_to_concat['yORIG_DTE']
        # concat to df
        df = pd.concat([df, batch_to_concat], axis=0)
        print('\nTotal number of rows of df: {0}'
              .format(df.shape[0]))

#%%
# change date format
df['PRD'] = lookup(df['PRD'])
# conversion to string speeds up merging below
df['strPRD'] = df['PRD'].astype(str)

print('\nCreating ORIG_data...')
# variables to include
ORIG_vars = (['strPRD', 'ORIG_AMT', 'ORIG_DTE', 'ORIG_YR', 'PRD', 'DID_DFLT'] +
             cat_vars + cont_vars)
# just get the last row for each LOAN_ID
ORIG_data = df.groupby('LOAN_ID')[ORIG_vars].last().reset_index()
print('Number of unique LOAN_IDs: {0}'.format(ORIG_data.shape[0]))

# delete DID_DFLT in original df
del df['DID_DFLT']

# get origination year
df['ORIG_YR'] = df['ORIG_DTE'].apply(lambda x: x.year)

# modify DID_DFLT column
to_merge = ORIG_data[['LOAN_ID', 'strPRD', 'DID_DFLT']]
df = df.merge(to_merge, how='left', on=['LOAN_ID', 'strPRD'])
# DID_DFLT is now 1 only on the date of default
df['DID_DFLT'] = df['DID_DFLT'].fillna(value=0)
# create default vars
df['DFLT_AMT'] = df['FIN_UPB'] * df['DID_DFLT']
df['NET_LOSS_AMT'] = df['NET_LOSS'] * df['DID_DFLT']

# delete unnecessary vars
del ORIG_data['strPRD']

#%%

# Questions
# For each column, show the number of missing values and percentage of total

# Calculate total number of loans for each year
# Calculate percentage of default for each origination year year

# Plot a histogram of net loss amounts for loans with
# nonzero losses in 2000, 2007 separately
# What percentage of loans in 2012, 2016 had nonzero losses?
