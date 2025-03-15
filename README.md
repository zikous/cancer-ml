## Overview
This project focuses on building a machine learning model to predict cancer. The research and development of the model are documented in the provided Jupyter notebook.

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

## License
This project is licensed under the MIT License.

