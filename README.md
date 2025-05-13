# 🧠 Sentiment Analysis on Twitter Data Using ML & DL Models

With the rapid growth of social media, platforms like Twitter (now X) have become powerful sources for understanding public sentiment. This project explores a range of classical machine learning and deep learning models to classify sentiment from tweet data, identifying the most effective model using the F1 score metric.

---

## 📌 Project Overview

This repository presents a comprehensive sentiment analysis pipeline built on labeled Twitter datasets. It includes preprocessing, feature extraction, model training, performance evaluation, and visualization.

**Key goals:**
- Compare classical ML (SVM, Random Forest, Naive Bayes) and DL (Feedforward NN, LSTM) models.
- Evaluate models based on F1 score.
- Apply the best model to domain-specific datasets for real-world insight.
- Visualize trends using Power BI.

---

## 🧰 Tools & Technologies

| Category       | Tools Used                        |
|----------------|-----------------------------------|
| ML Models      | SVM, Naive Bayes, Random Forest   |
| DL Models      | Feedforward Neural Network, LSTM  |
| Feature Extraction | Bag-of-Words, TF-IDF         |
| Visualization  | Power BI                          |
| Language       | Python                            |
| Libraries      | Scikit-learn, NLTK, Keras, Pandas |

---


---

## 📊 Results & Visualizations


### ✈️ Sentiment Analysis: Airlines Dataset

Using the best-performing model to analyze public sentiment toward airlines:

![Airline Sentiment](images/Airline Sentiment.png)

---

### 🌍 Sentiment by Geography: Squid Game on Netflix

Geographic sentiment distribution from Twitter data during the Squid Game trend:

![Squid Game Geo Sentiment](images/Show Review.png)

---

## 🔎 Key Insights

- **Classical ML models** like SVM and Random Forest offer fast, interpretable results when paired with BoW or TF-IDF.
- **LSTM models** captured nuanced text patterns, showing better performance on informal or emotional tweets.
- **Power BI** visualizations highlighted how sentiment shifts over time and geography.
- The **best model** was successfully transferred to analyze other domains like airline feedback and entertainment content.

---

📈 Future Enhancements
Incorporate transformer-based models (e.g., BERT)

Add real-time Twitter data streaming with Tweepy

Deploy as a sentiment analysis API or dashboard

Integrate additional metadata (hashtags, likes, retweets) for deeper analysis


🤝 Acknowledgments
Datasets from Kaggle and public Twitter archives

Libraries: Scikit-learn, Keras, NLTK, Matplotlib, Seaborn

Visualizations: Power BI



