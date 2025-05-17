#imports
from analysis.kaggle_analysis import load_kaggle_data, analyse_kaggle_data
from analysis.fnb_analysis import load_fnb_data, analyse_fnb_data
from utils.data_cleaning import clean_kaggle_data, clean_fnb_data
from utils.data_preprocessing import preprocess_kaggle_data, preprocess_fnb_data, get_top_kaggle_items, get_top_fnb_items
import os

#main function
def main():
    #Prompt user for dataset choice
    print("=== Dataset Analysis ===")
    print("1. Analyse Kaggle dataset")
    print("2. Analyse FNB dataset")
    print("3. Clean Kaggle dataset")
    print("4. Clean FNB dataset")
    print("5. Preprocess Kaggle Dataset")
    print("6. Preprocess FNB Dataset")
    print("7. Get Top 10 items from Kaggle dataset")
    print("8. Get Top 10 items from FNB dataset")
    choice = input("Choose dataset to analyze (1 - 8): ")

    #If user chooses Kaggle dataset
    if choice == "1":
        kaggle_path = os.path.join("data", "data.csv")
        df = load_kaggle_data(kaggle_path)
        analyse_kaggle_data(df)
    #If user chooses FNB dataset
    elif choice == "2":
        fnb_path = os.path.join("data", "dq_ps_challenge_v2 1.csv")
        df = load_fnb_data(fnb_path)
        analyse_fnb_data(df)
    #If user chooses to clean Kaggle dataset
    elif choice == "3":
        kaggle_path = os.path.join("data", "data.csv")
        df = load_kaggle_data(kaggle_path)
        cleaned_df = clean_kaggle_data(df)
        print("\nCleaned Kaggle dataset preview:")
        print(cleaned_df.head())
    #If user chooses to clean FNB dataset
    elif choice == "4":
        fnb_path = os.path.join("data", "dq_ps_challenge_v2 1.csv")
        df = load_fnb_data(fnb_path)
        cleaned_df = clean_fnb_data(df)
        print("\nCleaned FNB dataset preview:")
        print(cleaned_df.head())
    #If user chooses to preprocess Kaggle dataset
    elif choice == "5":
        kaggle_path = os.path.join("data", "data.csv")
        df = load_kaggle_data(kaggle_path)
        preprocessed_df, user_encoder, item_encoder = preprocess_kaggle_data(df)
        print("\nPreprocessed Kaggle dataset preview:")
        print(preprocessed_df.head())
    #If user chooses to preprocess FNB dataset
    elif choice == "6":
        fnb_path = os.path.join("data", "dq_ps_challenge_v2 1.csv")
        df = load_fnb_data(fnb_path)
        preprocessed_df, user_encoder, item_encoder = preprocess_fnb_data(df)
        print("\nPreprocessed FNB dataset preview:")
        print(preprocessed_df.head())
    #If user chooses to get top 10 items from FNB dataset
    elif choice == "7":
        kaggle_path = os.path.join("data", "data.csv")
        df = load_kaggle_data(kaggle_path)
        top_items = get_top_kaggle_items(df)
        print("\nTop 10 items from Kaggle dataset:")
        print(top_items)
    #If user chooses to get top 10 items from FNB dataset
    elif choice == "8":
        fnb_path = os.path.join("data", "dq_ps_challenge_v2 1.csv")
        df = load_fnb_data(fnb_path)
        top_items = get_top_fnb_items(df)
        print("\nTop 10 items from FNB dataset:")
        print(top_items)
    #If user chooses invalid option
    else:
        print("Invalid option.")

#Run the main function only if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()