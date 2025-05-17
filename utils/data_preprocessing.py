#imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Function to preprocess the kaggle data
def preprocess_kaggle_data(df):
    """
    Basic preprocessing for Kaggle dataset:
    - Encode CustomerID and StockCode
    - Return modified DataFrame and encoders
    """

    #Make a copy of the kaggle data
    df = df.copy()

    #Encode CustomerID and StockCode
    user_coder = LabelEncoder()
    item_coder = LabelEncoder()

    #Encode the CustomerID and StockCode as integers and store them in their new columns
    df['user_id'] = user_coder.fit_transform(df['CustomerID'])
    df['item_id'] = item_coder.fit_transform(df['StockCode'])

    #Basic interaction score mapping
    interaction_map = {'DISPLAY': 1, 'CLICK': 2, 'CHECKOUT': 3}
    df['interaction_score'] = df['Interaction'].map(interaction_map)

    return df, user_coder, item_coder