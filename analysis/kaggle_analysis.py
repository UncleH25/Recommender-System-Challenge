#imports
import pandas as pd
import os

#Dataset path
kaggle_dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')

#Function to load the dataset
def load_kaggle_data(kaggle_dataset_path):
    #Try to load the dataset
    try:
        df = pd.read_csv(kaggle_dataset_path, encoding='ISO-8859-1')
        return df
    #If there is an error, print the error message
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

#Function to analyze the dataset
def analyse_kaggle_data(df):
    print("\n===== Basic Info =====")
    print(df.info())

    print("\n===== Head =====")
    print(df.head())

    print("\n===== Description =====")
    print(df.describe(include='all'))

    print("\n===== Missing Values =====")
    print(df.isnull().sum())

    print("\n===== Unique Values per Column =====")
    print(df.nunique())

    #Function to clean the dataset
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