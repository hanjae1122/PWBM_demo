#!/usr/bin/env Rscript
if (!require("pacman")) install.packages("pacman")
pacman::p_load(optparse, # for reading command line input
               foreach, data.table, zoo, dplyr, ggplot2)

# create default command line args
option_list = list(
  	make_option(c("-s", "--cutoff_size"), type="integer", default=1000, 
              help="file reader will only read this many rows of each raw file [default=%default]", metavar="integer")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

## if (is.null(opt$cutoff_size)){
##   print_help(opt_parser)
##   stop("Cutoff size must be specified (using -s flag). Recommend 1000 for testing", call.=FALSE)
## }

fileslocation <- "raw"
outputlocation <- "clean"

# create directories
dir.create(file.path(outputlocation), showWarnings = FALSE)

# Create function to handle missing Current UPBs in the last record by setting them to the record prior
# forward fill
na.lomf <- function(x) {
  
  na.lomf.0 <- function(x) {
    non.na.idx <- intersect(which(!is.na(x)),which(x>0))
    if (is.na(x[1L]) || x[1L]==0) {
      non.na.idx <- c(1L, non.na.idx)
    }
    rep.int(x[non.na.idx], diff(c(non.na.idx, length(x) + 1L)))
  }
  
  dim.len <- length(dim(x))
  
  if (dim.len == 0L) {
    na.lomf.0(x)
  } else {
    apply(x, dim.len, na.lomf.0)
  }
}

Acquisitions_Variables = c(
  "LOAN_ID",
  "ORIG_CHN",
  "Seller.Name",
  "ORIG_RT",
  "ORIG_AMT",
  "ORIG_TRM",
  "ORIG_DTE",
  "FRST_DTE",
  "OLTV",
  "OCLTV",
  "NUM_BO",
  "DTI",
  "CSCORE_B",
  "FTHB_FLG",
  "PURPOSE",
  "PROP_TYP",
  "NUM_UNIT",
  "OCC_STAT",
  "STATE",
  "ZIP_3",
  "MI_PCT",
  "Product.Type",
  "CSCORE_C",
  "MI_TYPE",
  "RELOCATION_FLG"
)

Acquisition_ColClasses = c(
  "character",
  "character",
  "character",
  "numeric",
  "numeric",
  "integer",
  "character",
  "character",
  "numeric",
  "numeric",
  "character",
  "numeric",
  "numeric",
  "character",
  "character",
  "character",
  "character",
  "character",
  "character",
  "character",
  "numeric",
  "character",
  "numeric",
  "numeric",
  "character"
)

# Define Performance variables and classes, and read the files into R.
Performance_Variables = c(
  "LOAN_ID",
  "Monthly.Rpt.Prd",
  "Servicer.Name",
  "LAST_RT",
  "LAST_UPB",
  "Loan.Age",
  "Months.To.Legal.Mat",
  "Adj.Month.To.Mat",
  "Maturity.Date",
  "MSA",
  "Delq.Status",
  "MOD_FLAG",
  "Zero.Bal.Code",
  "ZB_DTE",
  "LPI_DTE",
  "FCC_DTE",
  "DISP_DT",
  "FCC_COST",
  "PP_COST",
  "AR_COST",
  "MHEC",
  "ATHP",
  "NS_PROCS",
  "CE_PROCS",
  "RMW_PROCS",
  "O_PROCS",
  "NON_INT_UPB",
  "PRIN_FORG_UPB",
  "REPCH_FLAG",
  "FCC_PRIN_WOA",
  "SRVC_ACT_IND"
)

Performance_ColClasses = c(
  "character",
  "character",
  "character",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "character",
  "character",
  "character",
  "character",
  "character",
  "character",
  "character",
  "character",
  "character",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "numeric",
  "character",
  "numeric",
  "character"
)

memory.size(100000)
memory.limit(100000)

# Check the number of files downloaded (should be even, equal number of Acquisition and Performance Files).
numberoffiles<-length(list.files(fileslocation, pattern = glob2rx("*txt"), full.names=TRUE))

# The "foreach" package contructs a loop so that R can iterate through all pairs of related Acquisition and Performance files.
# Calculate the number of iterations/cores in parallel processing allowing each pair to be processed simultaneously.
numberofloops<-(numberoffiles/2)

# Define Acquisition variables and classes, and read the files into R.
Acquisitions <- list.files(fileslocation, pattern = glob2rx("*Acquisition*txt"), full.names=TRUE)
Performance <- list.files(fileslocation, pattern = glob2rx("*Performance*txt"), full.names=TRUE)

# Pattern to extract Year and Quarter
pattern <- '[[:digit:]]{4}Q[[:digit:]]'
Years <- c()

# Ensure each pair downloaded for each quarter
for (k in 1:numberofloops) {
    acq_match <- regexpr(pattern, Acquisitions[k])
    acq_date <- regmatches(Acquisitions[k], acq_match)
    perf_match <- regexpr(pattern, Performance[k])
    perf_date <- regmatches(Performance[k], perf_match)
    if (acq_date == perf_date) {
        Years[k] <- acq_date
    } else {
        stop('Performance and Acquisition files not properly downloaded')
    }
}

for (k in 1:numberofloops) {
  print('=====================================================================')
  print(paste('Processing Data.A for', Years[k]))
  Data.A <- fread(Acquisitions[k], sep = "|",
                  colClasses = Acquisition_ColClasses, showProgress=FALSE)

  setnames(Data.A, Acquisitions_Variables)
  setkey(Data.A, "LOAN_ID")
  
  # Convert character variables to Date type
  # Data.A$ORIG_DTE<-as.yearqtr(Data.A$ORIG_DTE, "%m/%Y")
  
  # Delete unnecessary Acquisition variables.
  Data.A[,c("Seller.Name","Product.Type"):=NULL]
  
  # Obtain the Minimum Fico Score of the Borrower and Co-Borrower, Calculate House Price, and Replace Missing OCLTV values with OLTV values where available
  Data.A[, c("CSCORE_MN", "ORIG_VAL", "OCLTV"):= list(pmin(CSCORE_B,CSCORE_C, na.rm = TRUE),
                                                      (ORIG_AMT/(OLTV/100)),
                                                      ifelse(is.na(OCLTV), OLTV, OCLTV))]
  
  print(paste('Reading Data.P for', Years[k]))
  # Read and Process Performance data
  # ONLY READ subset of ROWS
  Data.P = fread(Performance[k], sep = "|", colClasses = Performance_ColClasses, 
                 showProgress=T, nrows=opt$cutoff_size)
  setnames(Data.P, Performance_Variables)
  
  print(paste('Converting dates for', Years[k]))
  # Convert character variables to Date type
  Data.P$Monthly.Rpt.Prd<-as.Date(Data.P$Monthly.Rpt.Prd, "%m/%d/%Y")
  Data.P$DISP_DT<-as.Date(Data.P$DISP_DT, "%m/%d/%Y")
  # Data.P$FCC_DTE<-as.Date(Data.P$FCC_DTE, "%m/%d/%Y")
  
  # Sort data by Loan ID and Monthly Reporting Period
  setorderv(Data.P, c("LOAN_ID", "Monthly.Rpt.Prd"))
  setkey(Data.P, "LOAN_ID")
  
  # REMOV LAST ORIG_DTE
  last.ID <- Data.P[dim(Data.P)[1], 'LOAN_ID']
  Data.P <- Data.P[!last.ID]
  
  # Standardize Delinquency Status Codes
  Data.P$Delq.Status<-as.numeric(ifelse(Data.P$Delq.Status %in% c("X", ""), "999", Data.P$Delq.Status))
  
  print(paste('Calculating loss vars for', Years[k]))
  # Vars created below only apply to last row of each loan
  # ex. 'Zero.Bal.Code', 'LPI_DTE', 'DISP_DT' are only available at last row of each loan
  # View(Data.P[1:100000, c('Zero.Bal.Code', 'LPI_DTE', 'DISP_DT')])
  
  # Define the last status of a loan and calculate the months between Last Paid Installment and Disposition date (for Lost Interest calculation)  
  Data.P[, c("LAST_STAT", "lpi2disp", "zb2disp"):= 
           list(ifelse(Zero.Bal.Code=='01', 'P',
                       ifelse(Zero.Bal.Code=='02', 'T',
                              ifelse(Zero.Bal.Code=='03', 'S', 
                                     ifelse(Zero.Bal.Code=='06', 'R', 
                                            ifelse(Zero.Bal.Code=='09', 'F', 
                                                   ifelse(Zero.Bal.Code=='15', 'N',
                                                          ifelse(Zero.Bal.Code=='16', 'L',
                                                                 ifelse(Delq.Status=='999','X',
                                                                        ifelse(Delq.Status >9, '9', 
                                                                               ifelse(Delq.Status==0, 'C', as.character(Delq.Status)
                                                                               )))))))))),
                ifelse(Data.P$LPI_DTE!="" & !(is.na(Data.P$DISP_DT)),as.numeric((year(DISP_DT)-year(as.yearmon(LPI_DTE, "%m/%d/%Y")))*12+month(DISP_DT)-month(as.yearmon(LPI_DTE, "%m/%d/%Y"))), 0),
                ifelse(!(is.na(Data.P$ZB_DTE)) & !(is.na(Data.P$DISP_DT)),as.numeric((year(DISP_DT)-year(as.yearmon(ZB_DTE, "%m/%Y")))*12+month(DISP_DT)-month(as.yearmon(ZB_DTE, "%m/%Y"))), 0)
           )]
  
  # View(Data.P[1:100000, c('Zero.Bal.Code', 'LPI_DTE', 'DISP_DT', 'ZB_DTE', "LAST_STAT", "lpi2disp", "zb2disp")])
  
  # create DID_DFLT
  # remove prepaid loans
  # remove loans that did not default but
  # zero-balanced before last date (<1% of loans)
  CreditEvents <- c("F", "S", "T", "N")
  Data.Last <- Data.P[, .SD[.N, .(LAST_STAT, Zero.Bal.Code)], by=LOAN_ID]
  Data.Last[, DID_DFLT := ifelse(LAST_STAT %in% CreditEvents, 1, 0)]
  Data.Last.Valid <- Data.Last[LAST_STAT != 'P' & (DID_DFLT != 0 | Zero.Bal.Code == "")]
  Data.P <- Data.P[LOAN_ID %in% Data.Last.Valid$LOAN_ID]
  
  # remove all but DID_DFLT
  Data.Last.Valid[, c("LAST_STAT", "Zero.Bal.Code") := NULL]
  
  # Add Original Rate from the Acquisitions Files
  Data.P[Data.A, ORIG_RT:=i.ORIG_RT, allow.cartesian=TRUE]
  
  # Apply function to backfill the first 6 missing entries in LAST_UPB
  #Data.P[, "LAST_UPB" :=na.locf(LAST_UPB, fromLast=T, maxgap=6, na.rm = FALSE), by = "LOAN_ID"]
  # Apply function to forwardfill (na.lomf forward fills but comment says backfill??) missing NON_INT_UPB and remaining missing current UPBs 
  # If c(NA,0,0,0), then becomes c(NA,NA,NA,NA)
  Data.P[, c("LAST_UPB", "NON_INT_UPB") :=list(na.lomf(LAST_UPB), na.lomf(NON_INT_UPB)), by = "LOAN_ID"]
  
  print(paste('Defining needed vars for', Years[k]))
  Data.P[, c("MODTRM_CHNG", "NON_INT_UPB", "PRIN_FORG_UPB", "MODUPB_CHNG"):= list(max(ifelse(length(unique(Maturity.Date))>1 & MOD_FLAG =="Y", 1, 0), 0, na.rm = TRUE),
                                                                                  -1*NON_INT_UPB,
                                                                                  -1*PRIN_FORG_UPB,
                                                                                  max(ifelse(!is.na(LAST_UPB) & !is.na(shift(LAST_UPB)) & MOD_FLAG =="Y" & LAST_UPB>shift(LAST_UPB), 1, 0), 0, na.rm = TRUE)), by = "LOAN_ID"]
  
  # Fin_UPB gives the final balance of the loans, so only applies to last row
  Data.P[, Fin_UPB := rowSums(.SD, na.rm = TRUE), .SDcols = c("LAST_UPB", "NON_INT_UPB", "PRIN_FORG_UPB")]
  
  Data.P[, c("modir_cost", "modfb_cost", "modfg_cost" ) := list(ifelse(MOD_FLAG =="Y", ((ORIG_RT - LAST_RT) / 1200) * LAST_UPB, 0),
                                                                ifelse(MOD_FLAG =="Y" & !is.na(NON_INT_UPB), -1*(LAST_RT / 1200) * NON_INT_UPB, 0),
                                                                ((-1*min(PRIN_FORG_UPB,0, na.rm = TRUE)) )), by = "LOAN_ID" ]
  Data.P[, c("C_modir_cost", "C_modfb_cost"):=list(cumsum(modir_cost),
                                                   cumsum(modfb_cost)), by = "LOAN_ID"]
  
  # Calculate Interest Cost, total expenses and total proceeds
  # View(Data.P[1:100000, c('FCC_COST','PP_COST','AR_COST','ATHP','MHEC')])
  # sapply(Data.P[, c('FCC_COST','PP_COST','AR_COST','ATHP','MHEC')], function(x) quantile(x, na.rm=T))
  
  # For non-last rows, "INT_COST","total_expense", "total_proceeds" are just zero
  Data.P[, c("INT_COST","total_expense", "total_proceeds") := 
           list(ifelse(LAST_STAT %in% CreditEvents & !is.na(DISP_DT), pmax(Fin_UPB *(((LAST_RT/100) - .0035)/12)*lpi2disp, 0),0),
                ifelse(LAST_STAT %in% CreditEvents & !is.na(DISP_DT), rowSums(Data.P[, list(FCC_COST,PP_COST,AR_COST,ATHP,MHEC)], na.rm = TRUE),0),
                ifelse(LAST_STAT %in% CreditEvents & !is.na(DISP_DT),(-1*rowSums(Data.P[, list(NS_PROCS,CE_PROCS,RMW_PROCS,O_PROCS)], na.rm = TRUE)),0))]
  # View(Data.P[1:100000, c('MHEC', 'NS_PROCS', "INT_COST","total_expense", "total_proceeds")])
  
  # Calculate Net Loss, Net Severity, Total Costs, Total Proceeds, and Total Liquidation Expenses.  Define Last Date variable.
  Data.P[,c("NET_LOSS","NET_SEV", "Total_Cost", "Tot_Procs", "Tot_Liq_Ex", "LAST_DTE"):=
           list(ifelse(LAST_STAT %in% CreditEvents & !is.na(DISP_DT), rowSums(Data.P[, list(LAST_UPB,INT_COST,total_expense,total_proceeds)], na.rm=TRUE),0),
                ifelse(LAST_STAT %in% CreditEvents & !is.na(DISP_DT), (rowSums(Data.P[, list(LAST_UPB,INT_COST,total_expense,total_proceeds)], na.rm=TRUE)/LAST_UPB),0),
                ifelse(LAST_STAT %in% CreditEvents, rowSums(Data.P[, list(LAST_UPB, INT_COST,FCC_COST,PP_COST, AR_COST, MHEC, ATHP)], na.rm = TRUE),0), 
                ifelse(LAST_STAT %in% CreditEvents, rowSums(Data.P[, list(NS_PROCS, CE_PROCS, RMW_PROCS, O_PROCS)], na.rm = TRUE),0),
                ifelse(LAST_STAT %in% CreditEvents, rowSums(Data.P[, list(FCC_COST, PP_COST, AR_COST, MHEC, ATHP)], na.rm = TRUE),0),
                as.Date(ifelse(!(is.na(Data.P$DISP_DT)), Data.P$DISP_DT, Data.P$Monthly.Rpt.Prd)))]
  # weirdly, some net_losses and thus net_sev are negative (ie. tot_procs > tot_cost)
  # View(Data.P[1:1000000, c('MHEC', 'NS_PROCS', "INT_COST","total_expense", "total_proceeds", "NET_LOSS","NET_SEV", "Total_Cost", "Tot_Procs", "Tot_Liq_Ex", "LAST_DTE")])
  
  # Delete Performance variables that are not needed.
  Data.P[, c("ZB_DTE", "ORIG_RT", "Servicer.Name", "lpi2disp"):=NULL]
  
  print(paste('Merging Data.A & Data.P for', Years[k]))
  # Merge together full Acquisition and Performance files.
  Combined_Loan_Data <- merge(Data.A, Data.Last.Valid, by.x = "LOAN_ID", by.y = "LOAN_ID", all = FALSE)
  Combined_Data = as.data.table(merge(Combined_Loan_Data, Data.P, by.x = "LOAN_ID", by.y = "LOAN_ID", all = FALSE))
  rm(Data.P, Data.A, Data.Last.Valid)
  
  # Calculate Modification Costs when loans default
  Combined_Data[,c("MODIR_COST","MODFB_COST"):=
                  list((ifelse((LAST_STAT %in% CreditEvents & !is.na(DISP_DT) & MOD_FLAG =="Y"),zb2disp*((ORIG_RT - LAST_RT) / 1200) * LAST_UPB, 0))+C_modir_cost,
                       (ifelse((LAST_STAT %in% CreditEvents & !is.na(DISP_DT) & !is.na(NON_INT_UPB) & MOD_FLAG =="Y"),zb2disp*(LAST_RT / 1200) * (-1*NON_INT_UPB), 0))+C_modfb_cost)]
  
  Combined_Data[, MODTOT_COST :=rowSums(.SD, na.rm = TRUE), .SDcols = c("modfg_cost", "MODIR_COST","MODFB_COST")]
  Combined_Data[,c("modir_cost", "modfb_cost"):=NULL]
  
  # remove unnecessary variables
  Combined_Data[,c("FRST_DTE", "OLTV", "CSCORE_B", "CSCORE_C", "LAST_RT", "LAST_UPB", "Loan.Age",
                   "Months.To.Legal.Mat", "Adj.Month.To.Mat", "Maturity.Date", 
                   "MSA", "Delq.Status", "MOD_FLAG", "Zero.Bal.Code", "LPI_DTE", 
                   "FCC_DTE", "DISP_DT", "FCC_COST", "PP_COST", "AR_COST", "MHEC", 
                   "ATHP", "NS_PROCS", "CE_PROCS", "RMW_PROCS", "O_PROCS", "NON_INT_UPB", 
                   "PRIN_FORG_UPB", "REPCH_FLAG", "FCC_PRIN_WOA", "SRVC_ACT_IND", 
                   "MODTRM_CHNG", "MODUPB_CHNG", "LAST_STAT", "zb2disp",
                   "modfg_cost", "C_modir_cost", "C_modfb_cost", "INT_COST", "total_expense", 
                   "total_proceeds", "NET_SEV", "Total_Cost", "Tot_Procs", 
                   "Tot_Liq_Ex", "LAST_DTE", "MODIR_COST", "MODFB_COST", "MODTOT_COST"):=NULL]
  
  # rename variable
  setnames(Combined_Data, c("Monthly.Rpt.Prd", "Fin_UPB"), c("PRD", "FIN_UPB"))
  
  # Save a Copy to disk or write a .txt file.
  year <- Years[k]
  print(paste('Exporting for', year))
  fwrite(Combined_Data, file = file.path(outputlocation, paste(opt$cutoff_size, year, 'csv', sep='.')))
  rm(Combined_Data)
}
