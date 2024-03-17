import pandas as pd

def load_txt(table_name):
    '''
    Load a table from a file and return a DataFrame.
    parameters: table_name (str) - the name of the file to read
    returns: DataFrame - the table
    '''
    return pd.read_table(table_name)