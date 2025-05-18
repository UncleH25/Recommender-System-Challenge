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