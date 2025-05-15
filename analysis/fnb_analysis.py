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