# Automatic Ticket Classification

## Overview
This project focuses on automating the classification of customer complaints using **Natural Language Processing (NLP)** techniques. The primary goal is to categorize support tickets based on the products and services mentioned, improving efficiency in customer support operations.

## Problem Statement
Financial companies receive numerous customer complaints daily, highlighting issues related to their products and services. Manually categorizing these complaints is time-consuming and inefficient. This project aims to automate the ticket classification process using machine learning, ensuring quicker response times and enhanced customer satisfaction.

## Business Goal
The objective is to develop a machine learning model capable of automatically classifying customer complaints into predefined categories. The solution leverages **Non-Negative Matrix Factorization (NMF)** for topic modeling and supervised classification models for automated ticket categorization.

## Categories for Classification
Customer complaints will be categorized into the following five groups:
1. **Credit card / Prepaid card**
2. **Bank account services**
3. **Theft/Dispute reporting**
4. **Mortgages/Loans**
5. **Others**

## Approach
The project follows a structured approach, divided into multiple steps:

### 1. Data Loading
- The dataset containing customer complaints is loaded in **JSON** format.
- The data is checked for missing values and inconsistencies.

### 2. Text Preprocessing
- Cleaning of text data (removal of stopwords, punctuation, special characters, etc.).
- Tokenization and lemmatization.
- Standardization of text for further analysis.

### 3. Exploratory Data Analysis (EDA)
- Analysis of word frequency and term distributions.
- Visualization of common words and themes using word clouds.
- Identification of patterns in customer complaints.

### 4. Feature Extraction
- **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Count Vectorization** are applied to convert text into numerical representations.

### 5. Topic Modelling
- **Non-Negative Matrix Factorization (NMF)** is used to extract latent topics from complaints.
- Topic distributions are analyzed and mapped to predefined complaint categories.

### 6. Model Building Using Supervised Learning
- Classification models such as **Logistic Regression**, **Decision Tree**, and **Random Forest** are trained on the labeled dataset.
- Comparison of different models to select the best-performing one.

### 7. Model Training and Evaluation
- Splitting the dataset into training and testing subsets.
- Training the model on preprocessed complaint data.
- Evaluating model performance using metrics such as **accuracy, precision, recall, and F1-score**.

### 8. Model Inference
- Deployment of the trained model for real-time classification of new customer complaints.
- Testing the model's generalization on unseen data.

## Results
- The model successfully classifies customer complaints into relevant categories.
- **NMF-based topic modeling** provides valuable insights into recurring customer issues.
- The trained classifier achieves high accuracy, making it suitable for practical implementation.

## Future Enhancements
- Incorporating **Deep Learning** models such as LSTMs and Transformer-based architectures.
- Fine-tuning preprocessing techniques for improved model performance.
- Expanding the dataset to handle multilingual complaints.

## Contributor
This project was developed by **Sagar Maru**. You can find more details on GitHub: [Sagar Maru](https://github.com/sagar-maru).