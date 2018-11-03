# FRED database
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

DATA_PATH = 'data/economic/raw'
EXPORT_PATH = 'data/economic'

# FRED data
# must be at least monthly
# (new_name, is_index)
DATA_DICT = {
    'USD1MTD156N': ('LIBOR', False),
    'CPIAUCSL': ('CPI', True),
    'INTDSRUSM193N': ('IR', False),
    'LRUN64TTUSM156S': ('UNEMP', False),
    'MORTGAGE30US': ('MR', False),
    'CSUSHPISA': ('HPI', True)
}

def read_FRED_data(filename):
    return pd.read_csv(filename, parse_dates=[0], na_values='.')

def get_to_concat(temp, new_name, is_index):
    temp['DATE'] = temp['DATE'].dt.to_period('m')
    grouped = temp.groupby('DATE')
    if is_index:
        firsts = grouped.first()
        return pd.Series((firsts[1:].values/firsts[:-1].values).ravel()-1, index=firsts.index[1:]).rename(new_name)
    else:
        return grouped.mean()/100

df = pd.DataFrame()

for name, v in DATA_DICT.items():
    new_name, is_index = v
    filename = os.path.join(DATA_PATH, '{0}.csv'.format(name))
    temp = read_FRED_data(filename)
    temp.columns = ['DATE', new_name]
    to_concat = get_to_concat(temp, new_name, is_index)
    print(to_concat.head())
    df = pd.concat([df, to_concat], axis=1, join='outer')

#Non-FRED
# must be at least monthly
# (new_name, is_index)
DATA_DICT2 = {
    'US-Monthly-GDP-History': ('rGDP', True)
}

def read_data(filename):
    return pd.read_csv(filename, parse_dates=[0])

for name, v in DATA_DICT2.items():
    new_name, is_index = v
    filename = os.path.join(DATA_PATH, '{0}.csv'.format(name))
    temp = read_data(filename)
    temp.columns = ['DATE', new_name]
    to_concat = get_to_concat(temp, new_name, is_index)
    print(to_concat.head())
    df = pd.concat([df, to_concat], axis=1, join='outer')

df_red = df['1993-01':]
df_red = df_red.reset_index()
df_red.plot()
plt.show()
df_red.to_csv(os.path.join(EXPORT_PATH, 'agg_ntnl_mnthly.csv'), index=False)



