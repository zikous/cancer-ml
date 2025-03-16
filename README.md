# Cervical Cancer Risk Prediction - Machine Learning Pipeline

## Overview
This project focuses on building an end-to-end machine learning pipeline to predict the risk of cervical cancer based on patient risk factors. The goal is to develop a robust and interpretable model that can assist healthcare professionals in early diagnosis and risk assessment. The pipeline includes data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation, culminating in a deployable machine learning model with enhanced explainability using SHAP.

The research and development process is documented in a Jupyter notebook, and the final model is deployed as a Streamlit web application for interactive use.

## Key Features

### Data Preprocessing
- **Handling Missing Values:** Replaces '?' with NaN and imputes missing values using the median.
- **Memory Optimization:** Reduces memory usage by downcasting numerical columns (e.g., float64 to float32, int64 to int32).
- **Class Imbalance:** Addresses class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

### Feature Selection
- **Correlation Analysis:** Selects features with an absolute correlation greater than a threshold (0.05) with the target variable (Biopsy).
- **Feature Importance:** Uses SHAP values to identify and visualize the most important features.

### Model Training
- **SHAP-Compatible Models:** Evaluates multiple machine learning algorithms compatible with SHAP explainability, including:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Extra Trees
- **Baseline Evaluation:** Trains and evaluates baseline models to identify the best-performing algorithm.

### Hyperparameter Tuning
- **RandomizedSearchCV:** Optimizes hyperparameters for each model using randomized search with cross-validation.
- **Best Model Selection:** Selects the best model based on the F1 score.

### Model Evaluation
- **Metrics:** Evaluates models using accuracy, precision, recall, F1 score, and ROC AUC.
- **Confusion Matrix:** Visualizes model performance on the test set.
- **ROC Curve:** Plots the Receiver Operating Characteristic curve to assess model performance.

### Explainability with SHAP
- **SHAP Summary Plot:** Visualizes feature importance.
- **SHAP Force Plot:** Explains individual predictions.
- **SHAP Dependence Plots:** Shows how top features affect predictions.

### Model Deployment
- **Streamlit App:** Deploys the best-performing model as an interactive web application.
  - Allows physicians to input patient data.
  - Displays predictions and SHAP explanations.
  - Provides actionable recommendations based on risk level.

## Repository Structure
- `notebook.ipynb`: Contains the Jupyter notebook with the entire machine learning pipeline.
- `model_artifacts/`:
  - `cervical_cancer_best_model.joblib`: The best-performing model.
  - `cervical_cancer_scaler.joblib`: The scaler used for feature scaling.
  - `feature_names.joblib`: The list of feature names used by the model.
- `data/`:
  - `risk_factors_cervical_cancer.csv`: The dataset used for training and testing.
- `requirements.txt`: Lists all the dependencies required to run the project.
- `app.py`: Streamlit application for interacting with the model.
- `cervical_cancer_pipeline.log`: Log file tracking the progress of the pipeline.

## Getting Started
Clone the repository:
```bash
git clone https://github.com/zikous/cancer-ml
```
Navigate to the project directory:
```bash
cd cancer-ml
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Explore the Notebook
Open `notebook.ipynb` to explore the data preprocessing, model training, and evaluation steps.

### Run the Streamlit App
Launch the Streamlit application:
```bash
streamlit run app.py
```
Input patient data in the sidebar to get predictions and SHAP explanations.

## Deployment
To deploy the Streamlit application, you can use platforms like:
- Heroku
- AWS
- Streamlit Sharing

Follow the respective platform's documentation for deployment instructions.

## Enhancements
- **Memory Optimization:** The `optimize_memory(df)` function reduces memory usage by downcasting numerical columns, making the pipeline more efficient for large datasets.
- **Cross-Validation:** Provides a more reliable estimate of model performance by evaluating models on multiple validation sets.
- **SHAP Explainability:** Offers interpretable insights into model predictions, making it easier to understand and trust the model.
- **Streamlit App:** Provides an intuitive interface for healthcare professionals to interact with the model.

## Future Work
- **Advanced Hyperparameter Tuning:** Explore techniques like Bayesian Optimization for hyperparameter tuning.
- **Feature Engineering:** Investigate additional feature engineering techniques to improve model performance.
- **Deployment Scaling:** Deploy the model on a cloud platform for scalability and accessibility.
- **User Interface Enhancements:** Improve the Streamlit app with additional features like patient risk profiling and visualizations.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to the branch.
4. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
The dataset used in this project is from the UCI Machine Learning Repository.

Special thanks to the open-source community for providing tools like Scikit-learn, XGBoost, LightGBM, and SHAP.
