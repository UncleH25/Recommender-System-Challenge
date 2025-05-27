# Recommender System Challenge

A comprehensive recommender system implementation that works with two different datasets: a Kaggle e-commerce dataset and an FNB (First National Bank) dataset. The project implements multiple recommendation approaches including popularity-based, collaborative filtering, and content-based filtering techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Recommendation Algorithms](#recommendation-algorithms)
- [Results](#results)
- [Requirements](#requirements)
- [License](#license)

## 🎯 Overview

This project demonstrates various recommendation system techniques applied to real-world datasets. It provides a modular approach to building, training, and evaluating different types of recommender systems, making it easy to compare their performance and understand their strengths and weaknesses.

## ✨ Features

- **Multiple Recommendation Approaches**: Popularity-based, collaborative filtering (user-based and item-based), and content-based filtering
- **Dual Dataset Support**: Works with both Kaggle e-commerce and FNB datasets
- **Data Pipeline**: Complete data cleaning, preprocessing, and analysis pipeline
- **Model Evaluation**: Built-in evaluation metrics including RMSE, Precision@K, and Recall@K
- **Interactive Interface**: Command-line interface for easy experimentation
- **Results Logging**: Automatic logging of model performance and results

## 📁 Project Structure

```
Recommender System Challenge/
├── analysis/
│   ├── fnb_analysis.py          # FNB dataset analysis functions
│   └── kaggle_analysis.py       # Kaggle dataset analysis functions
├── data/
│   ├── data.csv                 # Kaggle e-commerce dataset
│   └── dq_ps_challenge_v2 1.csv # FNB dataset
├── recommenders/
│   ├── collaborative_filtering_fnb.py    # Collaborative filtering for FNB
│   ├── collaborative_filtering_kaggle.py # Collaborative filtering for Kaggle
│   ├── content_based_fnb.py             # Content-based filtering for FNB
│   └── popularity_based.py              # Popularity-based recommendations
├── results/
│   ├── fnb_results.txt          # FNB model results log
│   └── kaggle_results.txt       # Kaggle model results log
├── utils/
│   ├── data_cleaning.py         # Data cleaning utilities
│   └── data_preprocessing.py    # Data preprocessing utilities
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Recommender System Challenge"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   For Kaggle-specific dependencies:
   ```bash
   pip install -r "requirements - kaggle.txt"
   ```

## 💻 Usage

Run the main application:

```bash
python main.py
```

The application provides an interactive menu with the following options:

1. **Analyse Kaggle dataset** - Explore the Kaggle e-commerce data
2. **Analyse FNB dataset** - Explore the FNB banking data
3. **Clean Kaggle dataset** - Apply data cleaning transformations
4. **Clean FNB dataset** - Apply data cleaning transformations
5. **Preprocess datasets** - Encode and prepare data for modeling
6. **Get Top 10 items** - Generate popularity-based recommendations
7. **Train models** - Train various recommendation algorithms
8. **Get recommendations** - Generate personalized recommendations
9. **Compare results** - Analyze user interactions vs recommendations

### Example Usage

```python
# Load and analyze data
from analysis.kaggle_analysis import load_kaggle_data, analyse_kaggle_data

df = load_kaggle_data("data/data.csv")
analyse_kaggle_data(df)

# Train a collaborative filtering model
from recommenders.collaborative_filtering_kaggle import train_kaggle_user_cf

model, precision, recall = train_kaggle_user_cf(preprocessed_df)
```

## 📊 Datasets

### Kaggle E-commerce Dataset
- **Source**: E-commerce transaction data
- **Key Features**: CustomerID, StockCode, Description, Quantity, UnitPrice, InvoiceDate
- **Use Case**: Product recommendation for online retail

### FNB Dataset
- **Source**: Banking/financial services interaction data
- **Key Features**: Customer ID, Item, Item Description, Interaction Type, Date
- **Interaction Types**: DISPLAY, CLICK, CHECKOUT
- **Use Case**: Financial product recommendation

## 🔧 Recommendation Algorithms

### 1. Popularity-Based Filtering
- **Kaggle**: Recommends items with highest total quantity sold
- **FNB**: Recommends most frequently checked-out items
- **Pros**: Simple, works for new users, good baseline
- **Cons**: No personalization, popularity bias

### 2. Collaborative Filtering

#### User-Based Collaborative Filtering
- Finds similar users based on interaction patterns
- Recommends items liked by similar users
- **Evaluation**: Precision@10: 1.0000, Recall@10: 0.7520

#### Item-Based Collaborative Filtering
- Finds similar items based on user interactions
- Recommends items similar to those the user has interacted with
- **Evaluation**: Precision@10: 0.8987, Recall@10: 0.7559

#### Matrix Factorization (ALS)
- Uses Alternating Least Squares for implicit feedback
- Handles sparse data effectively
- Implemented using the `implicit` library

### 3. Content-Based Filtering
- Uses TF-IDF vectorization on item descriptions
- Calculates cosine similarity between items
- Recommends items similar to user's interaction history
- **Features**: Item descriptions, categories, attributes

## 📈 Results

The system logs all training results and evaluations:

### Recent Performance Metrics
- **User-based CF**: Precision@10: 1.0000, Recall@10: 0.7520
- **Item-based CF**: Precision@10: 0.8987, Recall@10: 0.7559
- **ALS Model**: Successfully trained on FNB dataset

Results are automatically saved in the `results/` directory with timestamps.

## 🛠 Technical Details

### Data Processing Pipeline
1. **Data Loading**: Robust CSV loading with encoding handling
2. **Data Cleaning**: Remove nulls, handle cancellations, filter invalid entries
3. **Data Preprocessing**: Label encoding, interaction scoring, normalization
4. **Feature Engineering**: TF-IDF vectorization for content features

### Model Training
- **Surprise Library**: For traditional collaborative filtering
- **Implicit Library**: For matrix factorization with implicit feedback
- **Scikit-learn**: For content-based filtering and preprocessing

### Evaluation Metrics
- **RMSE**: Root Mean Square Error for rating prediction
- **Precision@K**: Proportion of relevant items in top-K recommendations
- **Recall@K**: Proportion of relevant items retrieved in top-K

## 📦 Requirements

### Core Dependencies
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `implicit` - Matrix factorization for recommender systems
- `scikit-surprise` - Collaborative filtering algorithms

### Optional Dependencies
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🔍 Future Enhancements

- [ ] Deep learning-based recommendations (Neural Collaborative Filtering)
- [ ] Hybrid recommendation systems
- [ ] Real-time recommendation API
- [ ] Web-based user interface
- [ ] A/B testing framework
- [ ] Cold start problem solutions
- [ ] Explainable recommendations

## 📞 Support

For questions, issues, or contributions, please:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Follow the contributing guidelines

---

**Note**: This project is designed for educational and research purposes. Ensure you have proper permissions to use the datasets in production environments.
