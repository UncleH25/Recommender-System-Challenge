#imports
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

#Function to build Interaction Matrix
def build_interaction_matrix(df, user_col='user_id', item_col='item_id', rating_col='interaction_score'):
    """
    Create a sparse user-item matrix for training
    """
    #Create a sparse matrix from the DataFrame
    return sparse.csc_matrix(
        (df[rating_col], (df[user_col], df[item_col]))
    )

#Function to train the ALS model
def train_implicit_als(interaction_matrix, factors=50, iterations=10, regularization=0.01):
    """
    Train an implicit ALS model
    """
    #Create the ALS model
    model = AlternatingLeastSquares(factors=factors,
                                     regularization=regularization,
                                     iterations=iterations)
    #Implicit expects item-user matrix
    model.fit(interaction_matrix.T)
    return model

#Function to log results into a file
def log_fnb_results(results, filename='fnb_results.txt'):
    """
    Log a message with a timestamp to the specified log file.
    """

    # Ensure the 'results' directory exists
    os.makedirs('results', exist_ok=True)
    
    #Get current timestamp in a readable format
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    #Write to the 'results' folder
    file_path = os.path.join('results', filename)

    #Open the log file in append mode
    with open(filename, 'a') as f:
        #Write the timestamp and results to the log file
        f.write(f"{timestamp} - {results}\n")