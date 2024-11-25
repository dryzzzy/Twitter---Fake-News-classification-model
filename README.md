# Twitter - Fake News Classification Model
Machine learning model (based on Natural Language Processing) for classification of tweets into two categories: true information or fake news.  
Dataset comes from [Kaggle: Fake News Detection](https://www.kaggle.com/datasets/jruvika/fake-news-detection/discussion?sort=hotness)

## **End Goal**
We will be building a solution based on NLP (Natural Language Processing).  
Our task will be to create a model that can classify text into one of two groups:
- True information (label 1)
- Fake information (fake news, label 2)

#### Technologies Used:
- **`pandas`**: Data manipulation, cleaning, and saving datasets.
- **`ydata-profiling`**: Generating exploratory data analysis profiling reports.
- **`nltk`**: Stopword removal and text tokenization.
- **`spacy`**: Lemmatization and advanced text processing.
- **`re`**: Regular expressions for text cleaning.
- **`gensim`**: For training the Word2Vec model and word vector analysis.
- **`sklearn`**: For training models, hyperparameter tuning, and classification reports.
- **`numpy`**: For numerical data manipulation and vector operations.

### **Best Model Evaluation Summary**
- **Best Model:** Support Vector Machine (SVM)
- **Best Parameters:** `C = 10`, `gamma = 'scale'`, `kernel = 'rbf'`
- **Key Performance Metrics:**
    - **Precision:** 0.98
    - **Recall:** 0.97
    - **F1-Score:** 0.98
    - **Accuracy:** 98%
    - **Weighted Average Precision:** 0.98
  
The **Support Vector Machine** model achieved the highest precision (0.98) and was chosen as the best-performing model.

---

## **Table of Contents**
1. [Notebook 1 - "1_data_preprocessing"](##-Notebook-1---"1_data_preprocessing")
2. [Notebook 2 - "2_vectors_report"](##-Notebook-2---"2_vectors_report")
3. [Notebook 3 - "3_model_training_and_tuning"](##-Notebook-3---"3_model_training_and_tuning")

---

## Notebook 1 - "1_data_preprocessing"

### 1. **Data Exploration & Profiling**
   - **Steps:**
     - Loaded the dataset from a CSV file.
     - Generated a profiling report for exploratory data analysis to understand the dataset distribution.
   - **Technologies Used:**
     - **`pandas`**: For data manipulation and analysis.
     - **`ydata-profiling`**: For generating a profiling report of the dataset.

### 2. **Data Filtering & Inspection**
   - **Steps:**
     - Filtered articles related to "politics" from the dataset for specific analysis.
     - Checked for missing values and duplicates in the dataset.
   - **Technologies Used:**
     - **`pandas`**: For data filtering and inspecting missing or duplicate entries.

### 3. **Data Cleaning**
   - **Steps:**
     - Dropped rows with missing values to ensure data consistency.
     - Combined the 'Headline' and 'Body' columns into a single feature for better context.
     - Removed unnecessary columns like 'URLs'.
   - **Technologies Used:**
     - **`pandas`**: For data cleaning and feature engineering.

### 4. **Text Preprocessing**
   - **Steps:**
     - Normalized text by converting it to lowercase and removing numbers and special characters.
     - Removed stopwords to reduce noise in the data.
     - Tokenized the text and performed lemmatization to standardize words.
   - **Technologies Used:**
     - **`nltk`**: For stopword removal and text tokenization.
     - **`spacy`**: For lemmatization and advanced text processing.
     - **`re`**: For regular expressions to clean unwanted characters.

---

## Notebook 2 - "2_vectors_report"

### 1. **Data Loading**
   - **Steps:**
     - Loaded the preprocessed dataset from a JSON file.
   - **Technologies Used:**
     - **`pandas`**: For loading the dataset into a 
