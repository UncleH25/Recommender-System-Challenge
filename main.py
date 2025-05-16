#imports
from analysis.kaggle_analysis import load_kaggle_data, analyse_kaggle_data, clean_kaggle_data
from analysis.fnb_analysis import load_fnb_data, analyse_fnb_data, clean_fnb_data
import os

#main function
def main():
    #Prompt user for dataset choice
    print("=== Dataset Analysis ===")
    print("1. Analyse Kaggle dataset")
    print("2. Analyse FNB dataset")
    print("3. Clean Kaggle dataset")
    print("4. Clean FNB dataset")
    choice = input("Choose dataset to analyze (1 - 4): ")

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
    #If user chooses invalid option
    else:
        print("Invalid option.")

#Run the main function only if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()