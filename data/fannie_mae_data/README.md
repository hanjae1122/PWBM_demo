# Preprocessing Fannie Mae

First, download the fannie mae data from [the site](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html) by clicking &lsquo;Access the Data&rsquo; on the right-hand side. A new account is required. We recommend downloading the &lsquo;Entire Single Family Eligible Fixed Rate Mortgage Dataset&rsquo;, which is about 24GB. You may initially download just a few quarters to ensure the code is functioning properly. Unzip all downloaded files. In each unzipped folder, there should be a pair of txt files, Acquisition and Performance, the names of which are in the following format:

    Acquisition_2001Q2.txt
    Performance_2001Q2.txt

Place all &lsquo;Acquistion&rsquo; and &lsquo;Performance&rsquo; files in &lsquo;raw&rsquo; in the data directory (&lsquo;fannie\_mae\_data&rsquo;), which you will need to create yourself:

    data/fannie_mae_data/raw

In order to preprocess the data, run the following line in the command line:

    Rscript --vanilla cleaning_initial_with_loss.R -s CUTOFF_SIZE

-   CUTOFF\_SIZE: is the number of rows that is read and processed. Reading the entire file may cause computational issues so we allow the option of limiting the filesize with this argument. We recommend using an initial cutoff size of 1000, which is the default, to ensure the code works.

## Output

The code automatically creates a new folder called &lsquo;clean&rsquo; in the data directory and places the preprocessed dataset in this folder.