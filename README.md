# 🧠 Diabetes Classification Using Machine Learning

Welcome to the project on **diabetes classification** using machine learning! 
🚀 This project aims to **predict** whether a patient **has diabetes or not** based on several health metrics, such as glucose levels, blood pressure, body mass index (BMI), and more.

## 👤 Author
[![GitHub](https://img.shields.io/badge/GitHub-ktzkvin-blue?logo=github&style=flat-square)](https://github.com/ktzkvin)  
[![Mail](https://img.shields.io/badge/Email-kevin.kurtz@efrei.net-blue?logo=gmail&style=flat-square)](mailto:kevin.kurtz@efrei.net)


## 📋 Features

- 🤖 **Machine Learning Models**: Implementation of several algorithms like Decision Tree, Logistic Regression, KNN, Multi-layer Perceptron (MLP), and Random Forest.

- 📊 **Model Evaluation**: Comparison of model performance using confusion matrices, classification reports, and accuracy scores.

- ⚙️ **Hyperparameter Tuning**: Optimization of model parameters with GridSearchCV to improve accuracy.

- 🔍 **Data Exploration**: Analysis of data distribution and handling of missing values using appropriate techniques (e.g., median imputation).

## 🚀 Installation and Setup

### Prerequisites

Ensure you have **Python 3.8+**. <br>
Required libraries: `pandas`, `numpy (1.26.0 or less)`, `scikit-learn`, `matplotlib`, `seaborn`, `ydata-profiling`.

### Installation

1. Clone the GitHub repository:

   ```bash
    git clone https://github.com/ktzkvin/Diabetes-Prediction-ML.git
    cd Diabetes-Prediction-ML

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

### Lunch the Notebook

Open and run the `Diabetes_Classification.ipynb` file to explore the project.

## 💾 Dataset

This project uses the **Pima Indians Diabetes Dataset**, available on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or on this repository as `data-diabetes.csv`. The dataset contains **768 records** of patients with **9 columns** (features like Glucose, BMI, Outcome, etc.).

**Outcome**: 1 for diabetic, 0 for non-diabetic.
Initial analysis shows a slightly **imbalanced dataset** (65% non-diabetic, 35% diabetic).

## 🔧 Preprocessing Steps
1. **Data Analysis**

    - **Class Distribution**: Visualizing the distribution of diabetic vs. non-diabetic patients.

    - **Data Profiling**: Using ydata-profiling to generate an interactive report and detect data anomalies.

2. **Handling Missing Values** 

Some columns (e.g., Glucose, BloodPressure) contain erroneous values (0). These values are replaced with the median of the data, calculated separately for each class (diabetic vs. non-diabetic), ensuring robustness for accurate predictions.

3. **Feature Selection and Normalization**

We use a correlation matrix to analyze feature redundancy and proceed with normalizing the data using StandardScaler before feeding it into machine learning models.

## 🤖 Machine Learning Models
The project explores several algorithms:

1. **Decision Tree** 🌳

2. **Logistic Regression** 📈

3. **KNN** 📍

4. **Multi-layer Perceptron (MLP)** 🧠

5. **Random Forest** 🌳🌲🌳

Each model is evaluated based on accuracy, precision and recall via confusion matrices and classification reports.

## 📈 Results

- **Best Model**: The Random Forest model achieved the best performance with an accuracy of **88.96%** after hyperparameter tuning.

- **Feature Importance Visualization**: Using the Random Forest model, we determined that the most important variables are Insulin, Glucose and SkinThickness.

#### ⚙️ Hyperparameter Tuning

We use GridSearch to find the best performance of the Decision Tree and Random Forest models by testing different parameter combinations.


## 🧠 Conclusions
In this project, we tested several machine learning models to predict diabetes, including Decision Trees, Logistic Regression, KNN, MLP, and Random Forest. The **Random Forest** model gave the best results, with an accuracy of **88.96%** after tuning the hyperparameters.

While Random Forest was the most accurate, simpler models like Decision Trees and Logistic Regression are still valuable because they are easier to interpret. Overall, this project shows how machine learning can be used to help predict diabetes based on health data.


## 🔧 Technologies Used
- **Jupyter Notebook** for interactive development.

- **Pandas** and **NumPy** for data manipulation.

- **Matplotlib** and **Seaborn** for data visualization.

- **Scikit-learn** for machine learning algorithms.

## 💡 Notes
Results may vary slightly depending on hyperparameters and data preprocessing methods. Iterative refinement and adjustments are recommended to further improve model performance.

---