#imports
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

#Function to train the user collaborative filtering model
def train_kaggle_user_cf(df, user_col='user_id', item_col='item_id', rating_col='Quantity'):
    """
    Train a user-based CF model using Kaggle dataset
    """
    #Create a Reader object with the rating scale from 1 to the maximum value in the rating column
    reader = Reader(rating_scale=(1, df[rating_col].max()))
    #Load the DataFrame into a Surprise Dataset using the specified user, item, and rating column
    data = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
    #Split the dataset into training and testing sets (80% train, 20% test)
    trainset, testset = train_test_split(data, test_size=0.2)

    #Initialise the KNNBasic algorithm for user-based collaborative filtering
    algo = KNNBasic(sim_options={'user_based': True})
    #Train the algorithm on the training set
    algo.fit(trainset)
    #Generate predictions on the test set
    predictions = algo.test(testset)
    #Print the Root Mean Squared Error (RMSE) of the predictions
    print("User-based CF RMSE (Kaggle):", accuracy.rmse(predictions))
    return algo