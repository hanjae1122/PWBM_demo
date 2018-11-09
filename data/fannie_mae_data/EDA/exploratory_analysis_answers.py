# Questions
# For each column, show the number of missing values and percentage of total
def missing_values(data):
    # getting the sum of null values and ordering
    total = data.isnull().sum().sort_values(ascending=False)
    # getting the percent and order of null
    percent = (data.isnull().sum() /
               data.isnull().count() * 100).sort_values(ascending=False)
    # Concatenating the total and percent
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
    print("Columns with at least one na")
    # Returning values of nulls different of 0
    print(df[~(df['Total'] == 0)])
    return
missing_values(ORIG_data)


# Calculate total number of loans for each year
# Calculate percentage of default for each origination year year
ORIG_data['ORIG_YR'] = ORIG_data['ORIG_DTE'].apply(lambda x: x.year)
ORIG_data['ORIG_YR'].value_counts()
ORIG_data.groupby('ORIG_YR')['DID_DFLT'].mean()

# Plot a histogram of net loss amounts for loans with
# nonzero losses in 2000, 2007 separately
# What percentage of loans in 2012, 2016 had nonzero losses?
net_loss_amts = df.groupby('LOAN_ID')['ORIG_YR', 'NET_LOSS_AMT'].last()

for yr, net_loss_in_yr in net_loss_amts.groupby('ORIG_YR'):
    if yr in [2000, 2007]:
        nonzero = net_loss_in_yr[net_loss_in_yr.NET_LOSS_AMT != 0]
        sample = nonzero.sample(1500, replace=True)
        plt.hist(sample['NET_LOSS_AMT'], alpha=0.5, label=str(yr))
plt.legend(loc='upper right')
plt.show()

for yr, net_loss_in_yr in net_loss_amts.groupby('ORIG_YR'):
    if yr in [2012, 2016]:
        print('For year {0}, {1:.4f}% had nonzero loss'
              .format(yr, (net_loss_in_yr.NET_LOSS_AMT != 0).mean()*100))
