# Bankruptcy Prediction Model

## Project Overview

This project aims to develop a predictive model that uses financial, liabilities, market, and asset parameters from a 10-year historical dataset to forecast the bankruptcy status of companies over the next five years. The model enhances decision-making for stakeholders and improves economic stability.

The target variable, `status_label`, classifies companies into either **'Alive'** or **'Failed'**, reflecting their bankruptcy status for the next 5 years.

App link - https://bankruptcy-wv2iiuquqqj9ozgmmjtofm.streamlit.app/

## Machine Learning Models Used

The project employs different models to predict the bankruptcy status of companies:

- **LSTM (Long Short-Term Memory)**
- **RNN (Recurrent Neural Networks)**
- **Transformers**

Among these, **LSTM** has provided the best results, achieving an accuracy of **94%**.

## Website Functionalities

The website built using **Streamlit** offers two key functionalities:

### 1. **Fiscal Stats Gallery**
The **Fiscal Stats Gallery** displays interactive visual representations using **Seaborn** and **Plotly** charts, providing insights into the following aspects of financial data:
- **Variation of Income Over Years**
- **Distribution of Total Operating Expenses**
- **Current Liabilities Comparison**
- **Comparison of Current Assets and Total Assets**

### 2. **Fiscal Collapse Check**
The **Fiscal Collapse Check** allows companies to upload their financial data (using the sample `test_a.csv` file from the repository) and forecast whether the company is likely to go bankrupt in the upcoming years. The model uses historical data to make predictions on the company's financial health.


