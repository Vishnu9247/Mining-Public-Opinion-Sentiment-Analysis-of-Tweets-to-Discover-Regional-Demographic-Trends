# Twitter Sentiment Analysis: Discovering Regional & Demographic Trends

![Data Mining](https://img.shields.io/badge/Data%20Mining-Sentiment%20Analysis-blue)
![Python ML](https://img.shields.io/badge/Python-Machine%20Learning-green)
![NLP](https://img.shields.io/badge/NLP-Text%20Analysis-orange)

## Overview

This project explores sentiment analysis on Twitter data to discover patterns in public opinion across different topics and regions. We developed and compared various machine learning and deep learning models to classify tweet sentiments, then applied the best performing model to analyze airline customer satisfaction and geographical distribution of opinions about the Netflix series "Squid Game."

## Team Members

**Team Data Smugglers**
- Vishnu Alla (Michigan Technological University)
- Vineeth Karjala (Michigan Technological University)
- Surya Vakkalagadda (Michigan Technological University)

## Objectives

- Identify the optimal sentiment classification model through comparison of various ML and DL techniques
- Demonstrate practical applications of sentiment analysis for large-scale public opinion mining
- Visualize sentiment trends across different entities (airlines) and geographic regions

## Datasets

| Dataset | Usage | Size | Input Feature | Target Variable |
|---------|-------|------|--------------|----------------|
| Dataset 1 | Training & Testing & Validation | 4,869 | Tweet text | Sentiment Label |
| Dataset 2 | Airline Sentiment Analysis | 14,640 | Tweet text | Inferred sentiment |
| Dataset 3 | Squid Game Sentiment by Location | 56,149 | Tweet text | Inferred sentiment |

## Methodology

### Data Preprocessing
- Text cleaning: Lowercase conversion, removal of URLs, user mentions, hashtags, and special characters
- Tokenization and lemmatization
- Stop word removal
- Label encoding for sentiment classes

### Feature Extraction
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Bag of Words (BoW)

### Models Developed & Evaluated
- Support Vector Machines (SVM)
- Random Forest
- Naïve Bayes
- Feedforward Neural Networks
- Long Short-Term Memory (LSTM) networks

## Results

The Feedforward Neural Network achieved the highest F1 score (0.646), followed closely by SVM with TF-IDF vectors (0.632). Due to its balance of performance and computational efficiency, the SVM model was selected for downstream analysis.

| Model | Input features | F1 Score | % Difference from Best Model |
|-------|---------------|----------|------------------------------|
| SVM | TF-IDF vectors | 0.632 | -2.17 |
| Random Forest | TF-IDF vectors | 0.615 | -4.8 |
| Naïve Bayes | TF-IDF vectors | 0.606 | -6.2 |
| SVM | BOW vectors | 0.616 | -4.64 |
| Random Forest | BOW vectors | 0.622 | -3.72 |
| Naïve Bayes | BOW vectors | 0.620 | -4.02 |
| Feedforward Neural Network | TF-IDF vectors | 0.646 | 0 |
| LSTM | TF-IDF vectors | 0.620 | -4.02 |

## Key Findings

### Airline Sentiment Analysis
- United Airlines had the highest volume of tweets with significant negative sentiment
- Virgin America showed better customer perception with higher proportion of positive tweets
- American, US Airways, and Southwest displayed similar sentiment profiles

### Geographic Sentiment for "Squid Game"
- North America and Western Europe showed predominantly positive sentiment
- Regional variations in sentiment were observed across different parts of the world
- Visualization enabled understanding of regional audience reception differences

## Technologies Used

- **Programming:** Python
- **ML Libraries:** Scikit-learn, TensorFlow/Keras
- **NLP Tools:** NLTK, spaCy
- **Visualization:** Power BI
- **Other Tools:** GeoText for location processing

## Getting Started

### Prerequisites
-Python 3.8+
-numpy
-pandas
-scikit-learn
-tensorflow
-nltk
-spacy
-geotext

### Installation

# Clone this repository
git clone https://github.com/datasmuggler/twitter-sentiment-analysis.git

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"

# Download spaCy model
python -m spacy download en_core_web_sm

### Project Structure

├── data/
│   ├── raw/               # Raw Twitter datasets
│   └── processed/         # Preprocessed data
├── models/                # Trained models
├── notebooks/            
│   ├── 1.0-preprocessing.ipynb
│   ├── 2.0-model-training.ipynb
│   └── 3.0-analysis.ipynb
├── src/                   # Source code
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── visualization/
├── visualizations/        # Power BI files and exports
├── requirements.txt
└── README.md
