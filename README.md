# **Online Shoppers Purchase Prediction Web App**

## Repository Outline
The file structure of this repository is as follows:

1. **README.md**                    - Overview of the project
2. **Notebooks**                   - Jupyter notebooks for EDA and model training
3. **Model Inference**              - Scripts for model inference and prediction
4. **app.py**                       - Streamlit application for EDA and prediction
5. **requirements.txt**             - List of Python dependencies used
6. **best_model_dtc.pkl**           - Best-trained Decision Tree model
7. **online_shoppers_intention.csv** - Main dataset in CSV format

## Problem Background
In e-commerce businesses, a high website visit rate does not always correlate with sales figures. Many visitors browse without making a purchase. Therefore, companies need to understand the characteristics and behaviors of users who are likely to make transactions. This project aims to build a predictive model based on user behavior data that can be used to predict the likelihood of purchase in real-time.

## Project Output
The output of this project is a Streamlit-based web application consisting of two main parts:

- Exploratory Data Analysis (EDA) to gain insights from user behavior.

- Prediction Page to predict the likelihood of purchase from user input based on the trained machine learning model.

## Data
The dataset used is the Online Shoppers Intention Dataset from the UCI Machine Learning Repository.

This dataset has:

- Number of rows: 12,330 observations

- Number of columns: 18 features + 1 target (Revenue)

- Data types: combination of numerical and categorical

- Target: Revenue (Boolean), whether the visitor made a purchase or not

- Before entering model training, the target column (Revenue) data type has been changed to integer

- There is an imbalance in the target class (Revenue = 15% True)

## Method
This project uses a supervised learning approach to solve the classification problem. Several models were tested, such as KNN, SVM, Random Forest, and Gradient Boosting. Evaluation results show that the Decision Tree Classifier has the highest and most stable F1 Score for the baseline model, and is chosen as the best model. The model is trained using the Scikit-Learn pipeline and saved in `.pkl` format using pickle.

## Stacks
Here are the technologies and tools used in this project:

- Programming Language: Python

    - Libraries:

        - `scikit-learn` (for modeling)

        - `pandas`, `numpy` (for data manipulation)

        - `matplotlib`, `seaborn` (for visualization)

        - `streamlit` (for building web apps)

        - Platform Deployment: Hugging Face Spaces

## Reference
- [UCI Machine Learning Repository – Online Shoppers Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- [Geeks for Geeks](https://www.geeksforgeeks.org/python/python-program-to-convert-camel-case-string-to-snake-case/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Huggingface](https://huggingface.co/spaces/yunidobaheramsyah/online-shoppers-intention)

---

**Additional References:**
- [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Contoh readme](https://github.com/fahmimnalfrzki/Swift-XRT-Automation)
- [Another example](https://github.com/sanggusti/final_bangkit) (**Must read**)
- [Additional reference](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)