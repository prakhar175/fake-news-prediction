# Fake News Prediction

A machine learning project that detects fake news using various classification algorithms and deep learning techniques. This project implements multiple approaches to identify misinformation and provides comparative analysis of different models' performance.

## 🎯 Project Overview

This project aims to classify news articles as real or fake using natural language processing and machine learning techniques. The system employs both traditional machine learning algorithms and deep learning models to achieve accurate fake news detection.

## 🔧 Technologies Used

- **Python 3.9**
- **Machine Learning Libraries:**
  - scikit-learn
  - pandas
  - numpy
- **Deep Learning:**
  - TensorFlow/Keras
- **Natural Language Processing:**
  - NLTK
  - TF-IDF Vectorization
- **Visualization:**
  - matplotlib
  - seaborn
  - wordcloud

## 📊 Models Implemented

### Traditional Machine Learning Models
1. **Multinomial Naive Bayes** - Probabilistic classifier based on Bayes' theorem
2. **Decision Tree** - Tree-based classification algorithm
3. **Random Forest** - Ensemble method using multiple decision trees

### Deep Learning Model
4. **GRU (Gated Recurrent Unit)** - Neural network for sequential data processing

## 🔄 Data Preprocessing

### For Traditional ML Models (Naive Bayes, Decision Tree, Random Forest):
- **Text Cleaning:** Removal of special characters, numbers, and unwanted symbols
- **Lowercasing:** Converting all text to lowercase
- **Stop Words Removal:** Filtering out common English stop words
- **TF-IDF Vectorization:** Converting text data into numerical features
- **Feature Extraction:** Creating term frequency-inverse document frequency vectors

### For Deep Learning Model (GRU):
- **Tokenization:** Converting text into sequences of tokens
- **Sequence Padding:** Ensuring uniform input length
- **Word Embeddings:** Converting tokens to dense vector representations

## 📈 Data Visualization

- **Word Clouds:** Visual representation of most frequent words in real vs fake news
  - Real news word cloud highlighting credible terminology
  - Fake news word cloud showing common misinformation patterns
- **Feature Analysis:** Distribution plots and statistical summaries
- **Model Performance Metrics:** Accuracy and loss for GRU


## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Multinomial Naive Bayes | 95.2270% |
| Decision Tree | 99.1260% |
| Random Forest | 98.1389% |
| GRU (Neural Network) | 99.8693% |

## 🎨 Visualizations

The project includes several visualization components:

- **Word Clouds:** Showing most frequent terms in real vs fake news
- **Model Comparison Charts:** Performance metrics visualization
- **Confusion Matrices:** Classification results analysis
- **Training History:** Loss and accuracy curves for the GRU model


## 🙏 Acknowledgments

- Dataset sources and contributors
- Open-source libraries and frameworks used
- Research papers and articles that inspired this project

⭐ If you found this project helpful, please give it a star!
