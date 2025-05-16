#imports
import pandas as pd

#Function to clean the kaggle dataset
def clean_kaggle_data(df):
    #Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    #Remove rows with missing values in 'CustomerID' and 'Description'
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    #Keep only rows with positive 'Quantity' and 'UnitPrice'
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    #Exclude rows where 'InvoiceNo' starts with 'C' (cancellations)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    #Convert 'InvoiceDate' to datetime format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

#Function to clean the fnb dataset
def clean_fnb_data(df):
    #Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    #Standardise column names with lower case and strip whitespace
    df.columns = df.columns.str.strip().str.lower()
    #Convert 'int_date' to datetime format, errors converted to NaT
    df['int_date'] = pd.to_datetime(df['int_date'], format='%d%b%Y', errors='coerce')
    df.dropna(subset=['int_date'], inplace=True)
    #Clean 'segment', 'beh_segment', and 'active_ind' columns. Convert to string, strip whitespace, and lower case
    df['segment'] = df['segment'].astype(str).str.strip().str.lower()
    df['beh_segment'] = df['beh_segment'].astype(str).str.strip().str.upper()
    df['active_ind'] = df['active_ind'].astype(str).str.strip().str.title()
    #Fill missing values in 'item_descrip' with 'None'
    df['item_descrip'] = df['item_descrip'].fillna('None')
    return df