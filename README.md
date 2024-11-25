# Twitter - Fake News Classification Model

Machine learning model (based on Natural Language Processing) for classification of tweets into two categories: true information or fake news.  
Dataset comes from [Kaggle: Fake News Detection](https://www.kaggle.com/datasets/jruvika/fake-news-detection/discussion?sort=hotness)

## **End Goal**
We will be building a solution based on NLP (Natural Language Processing).  
Our task will be to create a model that can classify text into one of two groups:
- True information (label 1)
- Fake information (fake news, label 2)

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

#### Technologies Used:
- **`pandas`**: Data manipulation, cleaning, and saving datasets.
- **`ydata-profiling`**: Generating exploratory data analysis profiling reports.
- **`nltk`**: Stopword removal and text tokenization.
- **`spacy`**: Lemmatization and advanced text processing.
- **`re`**: Regular expressions for text cleaning.
- **`gensim`**: For training the Word2Vec model and word vector analysis.
- **`sklearn`**: For training models, hyperparameter tuning, and classification reports.
- **`numpy`**: For numerical data manipulation and vector operations.

## **Table of Contents**
1. [Notebook 1 - "1_data_preprocessing"](##notebook-1---1_data_preprocessing)
2. [Notebook 2 - "2_vectors_report"](##notebook-2---2_vectors_report)
3. [Notebook 3 - "3_model_training_and_tuning"](##notebook-3---3_model_training_and_tuning)

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
     - **`pandas`**: For loading the dataset into a DataFrame.

### 2. **Tokenization**
   - **Steps:**
     - Tokenized the 'Headline+Body' column by splitting the text into individual words.
   - **Technologies Used:**
     - **`pandas`**: For applying tokenization on the dataset.

### 3. **Training Word2Vec Model**
   - **Steps:**
     - Trained a Word2Vec model on the tokenized text with specified hyperparameters.
     - Retrieved word vectors for analysis and explored word similarities (e.g., "politics", "trump").
   - **Technologies Used:**
     - **`gensim`**: For training the Word2Vec model and exploring word vectors.
     - **`numpy`**: For manipulating and calculating word embeddings.

### 4. **Creating Article Vectors**
   - **Steps:**
     - Calculated the average word vector for each article by averaging the word embeddings of the tokens in the article.
     - Added the calculated average vectors as new columns in the dataset.
   - **Technologies Used:**
     - **`numpy`**: For handling vector operations and calculating the average vectors for each article.

### 5. **Train/Test Split**
   - **Steps:**
     - Split the dataset into training and testing sets using an 80/20 ratio.
   - **Technologies Used:**
     - **`sklearn`**: For splitting the dataset into train and test sets.

### 6. **Model Training & Evaluation**
   - **Steps:**
     - Trained multiple classification models: Random Forest, Logistic Regression, and Support Vector Machine (SVM).
     - Generated classification reports to evaluate the performance of each model.
   - **Technologies Used:**
     - **`sklearn`**: For implementing and training classification models, as well as generating classification reports.

### 7. **Saving Data**
   - **Steps:**
     - Saved the dataset with added article vectors to a new JSON file for future use.
   - **Technologies Used:**
     - **`pandas`**: For saving the final dataset to a JSON file.

---

## Notebook 3 - "3_model_training_and_tuning"

### 1. **Data Loading**
   - **Steps:**
     - Loaded the dataset containing article vectors from a JSON file.
   - **Technologies Used:**
     - **`pandas`**: For loading the dataset into a DataFrame.

### 2. **Model Training & Hyperparameter Tuning**
   - **Steps:**
     - Prepared the feature matrix `X` (article vectors) and target variable `y` (labels).
     - Split the dataset into training and testing sets using an 80/20 ratio.
     - Used **GridSearchCV** for hyperparameter optimization on three models:
       - **Random Forest Classifier**
       - **Logistic Regression**
       - **Support Vector Machine (SVM)**
   - **Technologies Used:**
     - **`sklearn`**: For model training, hyperparameter tuning (GridSearchCV), and data splitting.

### 3. **Model Evaluation**
   - **Steps:**
     - Evaluated each model using the **classification report** to obtain precision, recall, and F1-score.
     - Printed the classification report for each model, showing the best-performing hyperparameters.
   - **Technologies Used:**
     - **`sklearn`**: For generating classification reports and evaluating model performance.

### 4. **Best Model Selection**
   - **Steps:**
     - Compared the models based on their **precision** scores from the classification report.
     - Selected the best model based on the highest precision value.
   - **Technologies Used:**
     - **`sklearn`**: For precision comparison and selecting the best-performing model.

### 5. **Result Analysis**
   - **Steps:**
     - Printed the best model and its corresponding precision value for final analysis.
   - **Technologies Used:**
     - **`sklearn`**: For result analysis and final model selection.
