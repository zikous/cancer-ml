# Cancer Prediction Machine Learning Project

## Overview
This project focuses on building a **machine learning model** to predict the likelihood of cervical cancer based on patient risk factors. The goal is to develop a robust and accurate model that can assist healthcare professionals in early diagnosis and risk assessment. The project includes data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation, culminating in a deployable machine learning pipeline.

The research and development process is documented in a Jupyter notebook, and the final model is deployed as a **Streamlit web application** for interactive use.

## Key Features
- **Data Preprocessing**: Handles missing values, scales features, and addresses class imbalance using SMOTE.
- **Feature Selection**: Uses correlation analysis to select the most relevant features for prediction.
- **Model Training**: Evaluates multiple machine learning algorithms, including Logistic Regression, Random Forest, XGBoost, LightGBM, and more.
- **Hyperparameter Tuning**: Optimizes model performance using RandomizedSearchCV.
- **Model Deployment**: The best-performing model is saved and deployed via a Streamlit app for real-time predictions.

## Repository Structure
- `notebook.ipynb`: Contains the Jupyter notebook with all the research and the process of creating the best model.
- `model_artifacts/`: Includes all the joblib files such as the model, scaler, and feature names.
- `data/`: Contains the dataset used for training and testing the model.
- `requirements.txt`: Lists all the dependencies required to run the project.
- `app.py`: Streamlit application for interacting with the model.

## Getting Started
1. Clone the repository:
    ```sh
    git clone /C:/Users/zakbh/Desktop/Projects/cancer-ml
    ```
2. Navigate to the project directory:
    ```sh
    cd cancer-ml
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Open the Jupyter notebook to explore the research and model development.
2. Use the joblib files in the `model_artifacts/` directory to load the model, scaler, and feature names for predictions.
3. Launch the Streamlit application:
    ```sh
    streamlit run app.py
    ```

## Deployment
To deploy the Streamlit application, you can use platforms like Heroku, AWS, or Streamlit Sharing. Follow the respective platform's documentation for deployment instructions.

