from SharedFunctions import *

DIR_INPUT='./data/'

BEGIN_DATE = "2018-04-01"
END_DATE = "2018-08-31"

print("Load  files")
transactions_df=read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE)
print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(transactions_df),transactions_df.TX_FRAUD.sum()))