# Diabetes Detection System 
Link : https://diabetes-detection-3ynjczd5ewdef9ghpqk8ad.streamlit.app/
## Overview
The Diabetes Detection System is a machine learning-based web application that predicts whether a person is diabetic based on various health factors. The system leverages a Support Vector Machine (SVM) classifier trained on the popular diabetes dataset, and it allows users to input data such as glucose levels, insulin levels, and other factors to predict the likelihood of diabetes.

The project includes:
- A machine learning model to classify diabetes status
- A user-friendly web interface for prediction
- Python-based implementation using libraries such as `sklearn`, `numpy`, and `streamlit`

## Features
- **Model Training**: The system uses the diabetes dataset to train an SVM classifier, achieving high accuracy on both training and testing datasets.
- **Prediction**: Users can input data about various health indicators, and the model will predict whether the individual is diabetic or not.
- **Web Application**: The system is built using `streamlit`, providing a simple interface for users to interact with.

## Requirements
Before running the project, ensure that you have the following libraries installed:
- `numpy`
- `pandas`
- `sklearn`
- `streamlit`

You can install these dependencies using `pip`:
```
pip install numpy pandas scikit-learn streamlit 
```

## Project Structure
```
Diabetes-Detection-System/
│
├── diabetes.csv             # Dataset file
├── trained_model.pkl        # Saved trained SVM model
├── Diabetes.ipynb           # Jupyter Notebook for training the model
├── diabetes_web.py          # Streamlit application
├── requirements.txt         # List of required dependencies
└── README.md                # This file
```

## How to Run the Project

1. **Train the Model**  
   First, run the `Diabetes.ipynb` Jupyter notebook to train the Support Vector Machine (SVM) model. The model is trained using the `diabetes.csv` dataset and then saved into a file named `trained_model.pkl` using pickle.

   You can open and run the notebook in a Jupyter environment (e.g., Jupyter Notebook, JupyterLab, or Google Colab).

2. **Start the Web App**  
   After training the model and saving it as `trained_model.pkl`, you can run the Streamlit web application. This will allow you to input data and receive diabetes predictions.

   ```bash
   streamlit run diabetes_web.py
   ```

   The web application will start, and you can visit it in your browser at `http://localhost:8501`.

3. **Input Data**  
   On the web interface, input the following health parameters for an individual:
   - Number of Pregnancies
   - Glucose Level
   - Blood Pressure value
   - Skin Thickness value
   - Insulin Level
   - BMI value
   - Diabetes Pedigree Function value
   - Age

   After entering the values, click the "Diabetes Test Result" button to get the prediction.

4. **View Prediction**  
   The app will display whether the person is diabetic or not based on the input data.

## Example

- **Input**:  
  - Pregnancies: 5  
  - Glucose: 166  
  - BloodPressure: 72  
  - SkinThickness: 19  
  - Insulin: 175  
  - BMI: 25.8  
  - Diabetes Pedigree Function: 0.587  
  - Age: 51

- **Output**:  
  "The person is diabetic"

These results show that the model is capable of predicting diabetes with a reasonable level of accuracy.

## Conclusion
This Diabetes Detection System provides a simple and effective way to predict whether an individual is at risk of diabetes. It uses machine learning techniques to make predictions based on various health indicators and provides an easy-to-use interface for users to input their data and receive immediate results.
