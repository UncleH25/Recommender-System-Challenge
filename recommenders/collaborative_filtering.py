#imports
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

#Function to build Interaction Matrix
def build_interaction_matrix(df, user_col='user_id', item_col='item_id', rating_col='interaction_score'):
    """
    Create a sparse user-item matrix for training
    """
    #Create a sparse matrix from the DataFrame
    return sparse.csr_matrix(
        (df[rating_col], (df[user_col], df[item_col]))
    )