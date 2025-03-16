# Cancer Prediction Machine Learning Project

## Overview
This project focuses on building a **machine learning model** to predict the likelihood of cervical cancer based on patient risk factors. The goal is to develop a robust and accurate model that can assist healthcare professionals in early diagnosis and risk assessment. The project includes data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation, culminating in a deployable machine learning pipeline.

The research and development process is documented in a Jupyter notebook, and the final model is deployed as a **Streamlit web application** for interactive use.

## Key Features
- **Data Preprocessing**:
  - Handles missing values, scales features, and addresses class imbalance using SMOTE.
  - **Memory Optimization**: Reduces memory usage by downcasting numerical columns (e.g., `float64` to `float32`, `int64` to `int32`).
- **Feature Selection**: Uses correlation analysis to select the most relevant features for prediction.
- **Model Training**: Evaluates multiple machine learning algorithms, including:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - Gradient Boosting
  - Neural Networks
  - K-Nearest Neighbors (KNN)
  - AdaBoost
  - Extra Trees
  - Balanced Bagging
- **Hyperparameter Tuning**: Optimizes model performance using `RandomizedSearchCV`.
- **Model Evaluation**:
  - **Cross-Validation**: Provides a more reliable estimate of model performance.
  - **Feature Importance Visualization**: Displays the importance of features for tree-based models.
  - **Learning Curves**: Diagnoses overfitting or underfitting by showing how model performance changes with increasing training data.
  - **ROC and Precision-Recall Curves**: Visualizes model performance, especially for imbalanced datasets.
- **Explainability**: Uses SHAP (SHapley Additive exPlanations) to interpret model predictions.
- **Model Deployment**: The best-performing model is saved and deployed via a Streamlit app for real-time predictions.
- **Logging**: Tracks the progress of the pipeline, especially for long-running tasks like hyperparameter tuning.

## Repository Structure
- `notebook.ipynb`: Contains the Jupyter notebook with all the research and the process of creating the best model.
- `model_artifacts/`: Includes all the joblib files such as:
  - `cervical_cancer_best_model.joblib`: The best-performing model.
  - `cervical_cancer_scaler.joblib`: The scaler used for feature scaling.
  - `feature_names.joblib`: The list of feature names used by the model.
  - `sample_input.joblib`: A sample input for testing the model.
- `data/`: Contains the dataset used for training and testing the model.
- `requirements.txt`: Lists all the dependencies required to run the project.
- `app.py`: Streamlit application for interacting with the model.
- `cervical_cancer_pipeline.log`: Log file tracking the progress of the pipeline.

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
1. Open the Jupyter notebook (`notebook.ipynb`) to explore the research and model development.
2. Use the joblib files in the `model_artifacts/` directory to load the model, scaler, and feature names for predictions.
3. Launch the Streamlit application:
    ```sh
    streamlit run app.py
    ```

## Deployment
To deploy the Streamlit application, you can use platforms like Heroku, AWS, or Streamlit Sharing. Follow the respective platform's documentation for deployment instructions.

## Enhancements
- **Memory Optimization**: The `optimize_memory(df)` function reduces memory usage by downcasting numerical columns, making the pipeline more efficient for large datasets.
- **Cross-Validation**: Provides a more reliable estimate of model performance by evaluating models on multiple validation sets.
- **Feature Importance Visualization**: Helps interpret the model by showing which features contribute most to predictions.
- **Learning Curves**: Diagnoses overfitting or underfitting by visualizing model performance as a function of training set size.
- **ROC and Precision-Recall Curves**: Provides a visual representation of model performance, especially for imbalanced datasets.
- **SHAP Explainability**: Offers interpretable insights into model predictions, making it easier to understand and trust the model.

## Future Work
- **Hyperparameter Optimization**: Explore advanced techniques like Bayesian Optimization for hyperparameter tuning.
- **Feature Engineering**: Investigate additional feature engineering techniques to improve model performance.
- **Deployment Scaling**: Deploy the model on a cloud platform for scalability and accessibility.
- **User Interface Enhancements**: Improve the Streamlit app with additional features like patient risk profiling and visualizations.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to the branch.
4. Submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The dataset used in this project is from the UCI Machine Learning Repository.
- Special thanks to the open-source community for providing tools like Scikit-learn, XGBoost, LightGBM, and SHAP.

---