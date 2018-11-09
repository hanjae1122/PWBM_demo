# import packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

sns.set(style="white", palette="muted", color_codes=True)

#%%
# SETTING PATH
#####################################################################
# YOU HAVE TO CHANGE THIS PATH SO IT POINTS TO YOUR LOCAL FOLDER
PATH = os.path.join(os.getcwd(), 'Downloads', 'project', 'PWBM_demo')
#####################################################################

ECON_PATH = os.path.join(PATH, 'data', 'economic')
CURR_PATH = os.path.join(PATH, 'data', 'fannie_mae_data', 'EDA')
CLEAN_PATH = os.path.join(PATH, 'data', 'fannie_mae_data', 'clean')
EXPORT_PATH = os.path.join(CURR_PATH, 'results')

# creates new folder at EXPORT_PATH
if not os.path.exists(EXPORT_PATH):
    print('Creating export folder')
    os.makedirs(EXPORT_PATH)

#%%
# READING ECON DATA
# parse_dates=['DATE'] reads in DATE as a np.datetime64 object
df_econ = pd.read_csv(os.path.join(ECON_PATH, 'agg_ntnl_mnthly.csv'),
                      parse_dates=['DATE'])

#%%
# explore econ data
df_econ.shape
df_econ.describe()
df_econ.columns

print('Econ df has dimension {0} and columns {1}'
      .format(df_econ.shape, ', '.join([c for c in df_econ.columns])))
# Note: [c for c in some_list] is called list comprehension

#%%
# PLOTTING
# initial series plot
df_econ.plot(x='DATE')

#%%
# series plots
date = df_econ['DATE']
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
for column in ['LIBOR', 'IR', 'UNEMP', 'MR']:
    plt.plot(date, df_econ[column])
plt.legend(loc='upper right')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
for column in ['CPI', 'HPI', 'rGDP']:
    plt.plot(date, df_econ[column])
plt.legend(loc='upper right')
plt.ylabel('Percent change')
plt.xlabel('Date')


# save plot as a file
# you have to run this together with the above block to get a file export
# do this by highlighting both blocks and running
plt.tight_layout()
plt.savefig(os.path.join(EXPORT_PATH, 'econ_series.png'), 
            bbox_inches='tight', dpi=300)

#%%
# correlation plots with seaborn
df_without_date = df_econ[['LIBOR', 'IR', 'UNEMP', 'MR', 'CPI', 'HPI', 'rGDP']]
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(df_without_date.corr())

#%%
# better correlation plot
# Generate a mask for the upper triangle
D = df_without_date.shape[1]
mask = np.zeros((D, D), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
plt.figure(figsize=(10, 10))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df_without_date.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()

#%%
# see cumulative growth of pct_chg values
pct_chg_df = df_econ[['DATE', 'CPI', 'HPI', 'rGDP']]
pct_chg_df.set_index('DATE', inplace=True)
# inplace=True makes is so that you don't have to set pct_chg_df again
# for example, without it we would write the above statement as
# pct_chg_df = pct_chg_df.set_index('DATE')
pct_chg_df = pct_chg_df.add(1)
pct_chg_df = pct_chg_df.cumprod()
pct_chg_df.dropna(inplace=True)

pct_chg_df.plot()

#%%
# READING LOAN DATA
# Don't worry too much about how 'with' works.
# It just allows you to temporarilty open the file to read it
# try and except is a way of catching errors.
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
                                    engine='c', parse_dates=['ORIG_DTE', 'PRD'])
            # know how .concat() works
            batch_to_concat = pd.concat([batch_to_concat, to_concat], axis=0)
        
        print('Total number of rows in batch: {0}'
              .format(batch_to_concat.shape[0]))
        
        year = yearlist[i]
        print('\nRemoving rows not in year {0}...'.format(year))

        # remove vintages not from the year in list   
        # .apply(f) applies a function f to a column
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

# make origination year column
df['ORIG_YR'] = df['ORIG_DTE'].apply(lambda x: x.year)
# reset index to range(df.shape[0])
df.reset_index(drop=True, inplace=True)

#%%
# .loc is a useful way to get subsets of a pandas df
# usage: df.loc[INDEX_RANGE, LIST_OF_COLUMNS]
# for example,
df.loc[100000:100050, ['LOAN_ID', 'ORIG_DTE', 'PRD', 'DID_DFLT', 'NET_LOSS']]

#%%
# IMPORTANT NOTE:
# If a loan defaulted, DID_DFLT = 1 for every PRD of that loan
# below we use .groupby() to set DID_DFLT to zero everywhere except the last period
# where it's 0 if the loan didn't default, and 1 if it did

# conversion to string speeds up merging below
# .astype converts to column to the specified type
df['strPRD'] = df['PRD'].astype(str)

print('\nCreating ORIG_data...')
# variables to include
ORIG_vars = (['strPRD', 'ORIG_AMT', 'ORIG_DTE', 'ORIG_YR', 'PRD', 'DID_DFLT'] +
             cat_vars + cont_vars)

# we use .groupby(), one of the most useful methods of pandas
# know how .groupby() is used.
# for each LOAN_ID, it lets us get the last row 
ORIG_data = df.groupby('LOAN_ID')[ORIG_vars].last().reset_index()
# thus, ORIG_data has one row for each LOAN_ID

print('Number of unique LOAN_IDs: {0}'.format(ORIG_data.shape[0]))

# delete DID_DFLT in original df
del df['DID_DFLT']

# modify DID_DFLT column
to_merge = ORIG_data[['LOAN_ID', 'strPRD', 'DID_DFLT']]
# .merge() is similar to .concat()
df = df.merge(to_merge, how='left', on=['LOAN_ID', 'strPRD'])

# DID_DFLT is now 1 only on the date of default and NA for the rest
# we fill those NA's with 0
df['DID_DFLT'] = df['DID_DFLT'].fillna(value=0)
# create AMT vars
df['DFLT_AMT'] = df['FIN_UPB'] * df['DID_DFLT']
df['NET_LOSS_AMT'] = df['NET_LOSS'] * df['DID_DFLT']
# the above 2 lines ensure we have DFLT_AMT, NET_LOSS_AMT 
# equal zero for all rows except the last PRD and
# equal zero for the last PRD if the loan did not default

# delete unnecessary vars
del ORIG_data['strPRD']

#%%
# merge econ vars to df
df_merged = df.merge(df_econ, how='left', left_on='PRD',
                     right_on='DATE', copy=False)

# DATE from df_econ is redundant with PRD
del df_merged['DATE']

#%%
# We have 3 dataframes we can use at this point
# - df: the original loan data
# - ORIG_data: data for each LOAN_ID
# - df_merged: df merged with economic variables


# Questions
# For each column in df_merged, show the number of missing values 
# and percentage of total that's missing.

# Calculate the total number of unique loans for each origination year
# Calculate percentage of defaulted loans for each origination year
# (suggestion: value_counts(), groupby())

# Plot a histogram of NET_LOSS_AMT for every loan originated in 2000, 2007 
# that had a nonzero NET_LOSS_AMT. 
# The 2000 and 2007 loans should be plotted on one plot but separately
# What percentage of loans originated in 2012, 2016 had nonzero NET_LOSS_AMT?

# Plot a useful graph that shows an interesting relationship between the economic variables and DID_DFLT. 
# You can be flexible in what variables and plot type (histogram, time series, correlation plot etc) you use.
libor = df_merged[['LIBOR']]
print('Number of missing values: {0}'.format(libor.isna().sum()))
print('percentage: {0}'.format(libor.isna().sum()/2239685))

cpi = df_merged[['CPI']]
print('Number of missing values: {0}'.format(cpi.isna().sum()))
print('percentage: {0}'.format(cpi.isna().sum()/2239685))

ir = df_merged[['IR']]
print('Number of missing values: {0}'.format(ir.isna().sum()))
print('percentage: {0}'.format(ir.isna().sum()/2239685))

unemp = df_merged[['UNEMP']]
print('Number of missing values: {0}'.format(unemp.isna().sum()))
print('percentage: {0}'.format(unemp.isna().sum()/2239685))

mr = df_merged[['MR']]
print('Number of missing values: {0}'.format(mr.isna().sum()))
print('percentage: {0}'.format(mr.isna().sum()/2239685))

hpi = df_merged[['HPI']]
print('Number of missing values: {0}'.format(hpi.isna().sum()))
print('percentage: {0}'.format(hpi.isna().sum()/2239685))

rgdp = df_merged[['rGDP']]
print('Number of missing values: {0}'.format(rgdp.isna().sum()))
print('percentage: {0}'.format(rgdp.isna().sum()/2239685))
#%%
ORIG_data['ORIG_YR'].value_counts()
#%%
a = df.loc[df['ORIG_YR']==2000]
b = a.loc[a['NET_LOSS_AMT']!=0]
c = b['NET_LOSS_AMT']
plt.subplot(211)
plt.hist(c)
d = df.loc[df['ORIG_YR']==2007]
e = d.loc[d['NET_LOSS_AMT']!=0]
f = e['NET_LOSS_AMT']
plt.subplot(212)
plt.hist(f)
#%%
df_2012 = df.loc[df['ORIG_YR']==2012]
net_loss_amt_2012 = df_2012[['NET_LOSS_AMT']]
nonz_net_loss_amt_2012 = net_loss_amt_2012.loc[net_loss_amt_2012['NET_LOSS_AMT']!=0]
print('Percentage of 2012: {0}'.format(nonz_net_loss_amt_2012.count()/net_loss_amt_2012.count()))

df_2016 = df.loc[df['ORIG_YR']==2016]
net_loss_amt_2016 = df_2016[['NET_LOSS_AMT']]
nonz_net_loss_amt_2016 = net_loss_amt_2016.loc[net_loss_amt_2016['NET_LOSS_AMT']!=0]
print('Percentage of 2016: {0}'.format(nonz_net_loss_amt_2016.count()/net_loss_amt_2016.count()))
