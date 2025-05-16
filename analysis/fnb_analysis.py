#imports
import pandas as pd
import os

#Dataset path
fnb_dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'dq_ps_challenge_v2 1.csv')

#Function to load the dataset
def load_fnb_data(fnb_dataset_path):
    # Try to load the dataset
    try:
        df = pd.read_csv(fnb_dataset_path)
        return df
    # If there is an error, print the error message
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

#Function to analyze the dataset
def analyse_fnb_data(df):
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
    df['item_descrip'].fillna('None', inplace=True)
    return df