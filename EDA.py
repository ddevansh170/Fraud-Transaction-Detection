from SharedFunctions import *
import autopep8

DIR_INPUT = './data/'

BEGIN_DATE = "2018-04-01"
END_DATE = "2018-08-31"

print("Load  files")
transactions_df = read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE)
print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(transactions_df),
                                                                               transactions_df.TX_FRAUD.sum()))

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 25)

print(transactions_df.head())

autopep8.fix_file("EDA.py")

print(transactions_df.shape)

print(transactions_df.dtypes)

print(transactions_df.isna().sum())

print(transactions_df.duplicated().sum())

print(transactions_df.info())



