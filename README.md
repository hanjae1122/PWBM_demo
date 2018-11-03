<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">CECL Project</a>
<ul>
<li><a href="#sec-1-1">Description</a></li>
<li><a href="#sec-1-2">Installation</a>
<ul>
<li><a href="#sec-1-2-1">Prerequisites</a></li>
<li><a href="#sec-1-2-2">Installing</a></li>
</ul>
</li>
<li><a href="#sec-1-3">Usage</a> 
<ul>
<li><a href="#sec-1-3-1">Project structure</a></li>
<li><a href="#sec-1-3-2">Inputs</a></li>
<li><a href="#sec-1-3-3">Vintage Aggregation</a></li>
<li><a href="#sec-1-3-4">Modeling</a></li>
</ul>
</li>
<li><a href="#sec-1-4">Notes</a></li>
<li><a href="#sec-1-5">Contributing</a></li>
<li><a href="#sec-1-6">Credits</a></li>
<li><a href="#sec-1-7">License</a></li>
</ul>
</li>
</ul>
</div>
</div>


# CECL Project<a id="sec-1" name="sec-1"></a>

## Description<a id="sec-1-1" name="sec-1-1"></a>

Given a portfolio of loans, we would like to model the net loss incurred over time. The final model output is lifetime expected credit losses for loss reserving under current expected credit loss (CECL). Broadly speaking, we use a probability of default model where expected loss is the product of 3 components:

<a href="https://www.codecogs.com/eqnedit.php?latex=\textrm{Net&space;loss}&space;=&space;\textrm{Probability&space;of&space;Default&space;}&space;*&space;\textrm{&space;Exposure&space;at&space;Default&space;}&space;*&space;\textrm{&space;Loss&space;given&space;Default&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textrm{Net&space;loss}&space;=&space;\textrm{Probability&space;of&space;Default&space;}&space;*&space;\textrm{&space;Exposure&space;at&space;Default&space;}&space;*&space;\textrm{&space;Loss&space;given&space;Default&space;}" title="\textrm{Net loss} = \textrm{Probability of Default } * \textrm{ Exposure at Default } * \textrm{ Loss given Default }" /></a>

We use Fannie Mae&rsquo;s [Single-Family Loan Performance Data](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html) to test our assumptions and model accuracy. The repository includes instructions and preprocessing code for the fannie mae dataset for those who wish to test the code on real data. The core code, however, will work with any generic loan dataset that fits the requirements described below.

## Installation<a id="sec-1-2" name="sec-1-2"></a>

### Prerequisites<a id="sec-1-2-1" name="sec-1-2-1"></a>

The main code base is written in python version 3.6 and relies on multiple packages including numpy, pandas and statsmodels. In order to keep track of package versions, we use [pipenv](https://pipenv.readthedocs.io/en/latest/), a virtual environment manager for python. It eases the process of setting up a working environment. Refer to its [documentation](https://pipenv.readthedocs.io/en/latest/install/) to see how it works.

The repository also includes preprocessing code for a dataset from Fannie Mae. Refer to the directory data/fannie\_mae\_data for more information. Preprocoessing code is written in R and also depends on various packages, which are specified in the file.

### Installing<a id="sec-1-2-2" name="sec-1-2-2"></a>

Clone this repository to your local machine

    git clone https://github.com/smetterspa/bank_model.git

Make sure you have python 3.6 and pipenv, which you can download using pip that comes with python.

    pip install pipenv

Change current directory to the cloned repository and use &lsquo;pipenv install&rsquo; to create a virtual environment and download all necessary packages simultaneously.

    cd path_to_cloned_repo
    pipenv install

Now you can run any script in the repository using pipenv.

    pipenv run python3 example.py

## Usage<a id="sec-1-3" name="sec-1-3"></a>

### Project structure<a id="sec-1-3-1" name="sec-1-3-1"></a>

    ├── models                  # Core code files
    │   ├── vintage             # Vintage analysis
    │   ├── individual          # Individual analysis (work in progress)
    ├── data                    # Contains all data needed for analysis
    │   ├── economic            # Macroeconomic data
    │   │   ├── *.csv           # Preprocessed macro data
    │   │   ├── *.py            # Preprocessing code
    │   │   └── raw             # Raw data
    │   ├── fannie_mae_data     # Fannie mae data
    │   │   ├── *.R             # Preprocessing code
    │   │   ├── raw             # Data before any preprocessing
    │   │   ├── clean           # Data after preprocessing
    │   │   └── README.md       # Instructions for cleaning Fannie Mae data
    ├── modules                 # Auxiliary code for main code
    ├── summary_notes.docx      # Contains detailed notes about project
    ├── Pipfile
    ├── Pipfile.lock
    └── README.md

### Inputs<a id="sec-1-3-2" name="sec-1-3-2"></a>

In order to ensure everything has been setup properly, we recommend running the code with the fannie mae dataset. Instructions for obtaining and preprocessing the fannie mae dataset are in &lsquo;data/fannie\_mae\_data&rsquo;. To use a custom dataset, make sure that it is in the required input format described below. We will follow the naming conventions of the fannie mae dataset, which are described [here](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html) under &lsquo;File Layout&rsquo;.

Required columns:
-   LOAN\_ID: A unique identifier for each individual loan
-   ORIG\_AMT: The original amount of the loan
-   PRD: Current month and year
-   ORIG\_DTE: Month and year loan was issued
-   FIN\_UPB: Outstanding unpaid principal balance of loan on the month of default
-   NET\_LOSS: Amount that was lost due to defaulted loan

Optional loan-specific columns:

[Categorical]
-   ORIG\_CHN: Origination Channel - Channel refers to three options: Retail (R), Correspondent (C), Broker (B)
-   PURPOSE: Loan Purpose - An indicator that denotes w hether the mortgage loan is either a refinance mortgage or a purchase money mortgage. Purpose may be the purchase of a new property or refinance of an existing lien (w ith cash out or with no cash out).
-   PROP\_TYP: Property Type - A field that denotes w hether the property type securing the mortgage loan is a cooperative share, condominium, planned urban development, single-family home, or manufactured home.
-   OCC\_STAT: Occupancy Status - An indicator that denotes w hether the mortgage loan, at its origination date, is secured by a principal residence, second home or investment property.
-   MI\_TYPE: Mortgage Insurance Type - The entity that is responsible for the Mortgage Insurance premium payment.
-   RELOCATION\_FLG: Relocation Mortgage Indicator - An indicator that denotes w hether or not the type of mortgage loan is a relocation mortgage loan, made to borrow ers w hose employers relocate their employees.

[Numerical]
-   ORIG\_RT: Original Interest Rate - The original interest rate on a mortgage loan as identified in the original mortgage loan documents.
-   ORIG\_TRM: Original Loan Term - The number of months in w hich regularly scheduled borrow er payments are due under the terms of the related mortgage documents.
-   OCLTV: Original Combined Loan-to-Value - A ratio calculated at the time of origination for a mortgage loan. The CLTV reflects the loan-to-value ratio inclusive of all loans secured by a mortgaged property on the origination date of the underlying mortgage loan.
-   NUM\_BO: Number of Borrowers - The number of individuals obligated to repay the mortgage loan.
-   DTI: Debt to Income Ratio - A ratio calculated at origination derived by dividing the borrower’s total monthly obligations (including housing expense) by his or her stable monthly income. This calculation is used to determine the mortgage amount for w hich a borrower qualifies.
-   NUM\_UNIT: Number of Units - The number of units comprising the related mortgaged property.
-   MI\_PCT: Mortgage Insurance Percentage
-   CSCORE\_MN: Credit Score at Origination - A numerical value used by the financial services industry to evaluate the quality of borrow er credit. Credit scores are typically based on a proprietary statistical model that is developed for use by credit data repositories. These credit repositories apply the model to borrow er credit information to arrive at a credit score. When this term is used by Fannie Mae, it is referring to the &ldquo;classic&rdquo; FICO score developed by Fair Isaac Corporation.

If using a custom dataset, create a new folder in &lsquo;data&rsquo; with the following structure.

    ├── data                    # Contains all data needed for analysis
    │   ├── custom_data         # Name of custom dataset
    │   │   ├── raw             # Data before any preprocessing
    │   │   └── clean           # Data after preprocessing
    └── ...

Place the original dataset in &lsquo;raw&rsquo; and the dataset that has been formatted according to the above specifications in &lsquo;clean&rsquo;. Specify the custom directory name when running code in the core code base. Whether using a custom dataset or the fannie mae dataset, from hereon, we will refer to the directory with the files as &lsquo;data directory&rsquo;.

### Vintage Aggregation<a id="sec-1-3-3" name="sec-1-3-3"></a>

Vintage level analysis requires loans to be aggregated by origination month. Loan information such as credit score will be weight-averaged by original amount of the loan. The aggregated data will be placed in the directory called &lsquo;vintage\_data&rsquo; within the data directory. To preprocess the data, run the following in command line:

    usage: vintage_aggregate.py [-h] [-l LOWER_LIM]
                                [dataname] [filelist] [varlist]
    
    Aggregate loan data by vintage.
    
    positional arguments:
      dataname              name of data folder (default: fannie_mae_data)
      filelist              text file containing list of file batches to read
                            (default: filelist.txt)
      varlist               text file containing a row of categorical and a row of
                            continuous loan variables (default: varlist.txt)
    
    optional arguments:
      -h, --help            show this help message and exit
      -l LOWER_LIM, --lower_lim LOWER_LIM
                            minimum number of loans per vintage (default: 500)

-   filelist: A textfile specifying file batches. The first element of each row must be the year for the batch. The remaining elements are the filenames in that batch. One batch will be read at a time. Place this file in the data directory.
-   varlist: A textfile specifying the optional loan variables. The first element of each row must be the keyword &lsquo;CAT&rsquo; for categorical variables or &lsquo;CONT&rsquo; for continuous variables. The remaining elements are the names of the columns.

A simple execution of aggregating the fannie mae dataset using pipenv will look like the following:

    pipenv run python vintage_aggregate.py -l LOWER_LIM

Running the code will create a new directory in the data directory called &lsquo;vintage\_analysis&rsquo;. Vintage aggregated data will be placed inside this directory under &lsquo;data&rsquo;.

    ├── data                    # Contains all data needed for analysis
    │   ├── data directory      # Name of custom dataset
    │   │   ├── raw             # Data before any preprocessing
    │   │   ├── clean           # Data after preprocessing
    │   │   ├── vintage_analysis
    │   │   │   ├── data        # Contains vintage-aggregated data
    │   │   │   └── results     # Results from running vintage modelling code
    └── ...

### Modeling<a id="sec-1-3-4" name="sec-1-3-4"></a>

With the dataset in the format we want, run the following to get model results:

    usage: vintage_analysis.py [-h] [dataname] filename
    
    Probability of default model for vintage data.
    
    positional arguments:
      dataname    name of data folder (default: fannie_mae_data)
      filename    name of data file
    
    optional arguments:
      -h, --help  show this help message and exit

A simple execution with pipenv will look like:

    pipenv run python vintage_analysis.py FILENAME

-   Model specifications

-   Output

    All results such as plots that are generated will be placed in &lsquo;vintage\_analysis&rsquo; under a newly created directory called &lsquo;results&rsquo;.

## Notes<a id="sec-1-4" name="sec-1-4"></a>

## Contributing<a id="sec-1-5" name="sec-1-5"></a>
- John Han: hanjae1122
- Yijie Gui: guiyijie
- Tong Shao: shaot0510
- Siyan Shen: siyanshen

## Credits<a id="sec-1-6" name="sec-1-6"></a>

## License<a id="sec-1-7" name="sec-1-7"></a>
