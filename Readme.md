
---

## 📝 Project Overview

This project analyzes public sentiment on Twitter with two main objectives:

1. **Airline Sentiment Analysis** – Understand customer opinions toward major airlines.
2. **Geographic Sentiment Mapping** – Analyze global sentiment around the TV show *Squid Game*.

We trained sentiment classification models using a labeled Twitter dataset and applied them on new datasets to uncover sentiment patterns.

---

## 🧹 Data Preprocessing

Implemented in: `Data Pre-Processing.ipynb`

- Lowercasing text
- Removing URLs, mentions, hashtags, special characters
- Tokenization and Lemmatization
- Stopword removal
- Label Encoding (Negative: -1, Neutral: 0, Positive: 1)
- Location filtering (for Squid Game tweets)
- Feature vectorization using Bag of Words and TF-IDF

---

## 🤖 Model Training

Implemented in:

- `Sentiment analysis models (Machine Learning).ipynb`
- `Deep Learning models (Sentiment Analysis).ipynb`

Models used:
- Logistic Regression
- Naive Bayes
- SVM
- Random Forest
- Feedforward Neural Network
- LSTM

Performance evaluated using accuracy and F1-score.

---

## 🔬 Experiments

### Experiment 1: Airline Sentiment Distribution  
**Notebook**: `Airlines.ipynb`

- Predicted sentiment on 14,640 airline-related tweets.
- Visualized using a 100% stacked column chart in Power BI.
- Observed customer satisfaction trends per airline.

### Experiment 2: Squid Game Sentiment by Location  
**Notebook**: `Squid Game.ipynb`

- Applied the model on 56,149 tweets mentioning *Squid Game*.
- Mapped predicted sentiment by user location using Power BI maps.
- Analyzed geographic variations in sentiment.

---

## 📊 Visualizations

Located in: `BI files/`

- **Airlines Sentiment.pbix** – Bar chart visualization for airline sentiment comparison.
- **Squid Game Sentiment Map.pbix** – Geographic sentiment distribution of tweets.

---

## 📄 Report

The final report summarizing methodology, model performance, and key findings can be found in [`ADM_Project.pdf`](./ADM_Project.pdf).

---

## 🛠️ Technologies Used

- Python (scikit-learn, NLTK, spaCy, Keras, TensorFlow)
- Jupyter Notebooks
- Power BI
- pandas, NumPy, Matplotlib

---

## 📬 Contact

For any questions or collaboration ideas, feel free to reach out via GitHub Issues or email.

---

