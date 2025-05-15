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