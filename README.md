Perfect ✅ — here’s your **final, polished `README.md`**, now including a well-organized **`requirements.txt` section** at the end. This version is formatted cleanly for GitHub and ready for direct use.

---

# 🏠 House Price Prediction Using Deep Learning (ANN)

## 🚀 How to Run the Project

Follow these steps to set up and run the project on your local machine:

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/House-Price-Prediction-Using-Deep-learning-ANN.git
cd House-Price-Prediction-Using-Deep-learning-ANN
```

### **2. Create and Activate Virtual Environment**

```bash
python -m venv .venv
```

**Activate the environment:**

* **Windows:**

  ```bash
  .venv\Scripts\activate
  ```
* **macOS/Linux:**

  ```bash
  source .venv/bin/activate
  ```

### **3. Install Required Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run the Flask Application**

```bash
python app.py
```

### **5. Access the Web App**

Open your browser and visit:

```
http://127.0.0.1:5000
```

You can now input house details and get instant price predictions.

---

## 📘 Table of Contents

1. [Introduction](#1-introduction)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Neural Network Architecture](#3-neural-network-architecture)
4. [Model Training](#4-model-training)
5. [Model Evaluation](#5-model-evaluation)
6. [Web Application using Flask](#6-web-application-using-flask)
7. [Documentation for Flask App](#7-documentation-for-flask-app)
8. [Requirements](#8-requirements)

---

## 1. Introduction

The **House Price Prediction System** predicts house prices based on various input features such as location, number of rooms, house age, and other amenities.
It uses an **Artificial Neural Network (ANN)** built with **Keras** and deployed through a **Flask** web application for user interaction.

---

## 2. Data Preprocessing

* **Data Scaling:**
  All features are normalized using **MinMaxScaler** from `scikit-learn` to ensure consistent input ranges for the neural network.
* **Data Splitting:**
  The dataset is divided into **training** and **testing** sets to evaluate model performance and prevent overfitting.

---

## 3. Neural Network Architecture

The ANN model architecture is structured as follows:

| Layer Type     | Neurons | Activation | Notes              |
| -------------- | ------- | ---------- | ------------------ |
| Input Layer    | 1000    | ReLU       | Input features     |
| Dropout Layer  | -       | -          | Dropout rate = 20% |
| Hidden Layer 1 | 500     | ReLU       | Fully connected    |
| Dropout Layer  | -       | -          | Dropout rate = 20% |
| Hidden Layer 2 | 250     | ReLU       | Fully connected    |
| Output Layer   | 1       | Linear     | Regression output  |

This architecture balances model capacity with regularization to reduce overfitting.

---

## 4. Model Training

* **Optimizer:** RMSprop
* **Loss Function:** Mean Squared Error (MSE)
* **Early Stopping:** Implemented with a patience of 50 epochs
* **Epochs:** 10
* **Batch Size:** 50

Training is monitored for validation loss to ensure the best generalization performance.

---

## 5. Model Evaluation

Model performance is evaluated using these metrics:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Mean Squared Logarithmic Error (MSLE)**
* **R-squared (R²) Score**

These metrics provide a detailed view of both absolute and relative prediction accuracy.

---

## 6. Web Application using Flask

A lightweight **Flask web application** is developed to make predictions using the trained model.

### Key Features:

* Interactive web form for user inputs
* Real-time prediction output
* Integration with pre-trained ANN and scaler
* Clean and simple UI

### Project Structure:

```
House-Price-Prediction-Using-Deep-learning-ANN/
│
├── app.py                # Flask main application
├── model.h5              # Trained ANN model
├── scaler.pkl            # MinMaxScaler object
├── templates/
│   └── index.html        # Front-end HTML page
├── static/
│   ├── css/
│   └── js/
├── data/
│   └── housing.csv       # Dataset (optional)
└── requirements.txt      # Dependencies list
```

---

## 7. Documentation for Flask App

The Flask app workflow:

1. **Load** pre-trained model (`model.h5`) and scaler (`scaler.pkl`) at startup.
2. **Accept** user inputs (longitude, latitude, rooms, age, etc.) from the form.
3. **Preprocess** the inputs using the loaded scaler.
4. **Predict** the house price using the trained ANN model.
5. **Display** the predicted price dynamically on the results page.

---

## 8. Requirements

Below is a sample `requirements.txt` for your project:

```txt
Flask==3.0.3
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
tensorflow==2.16.1
keras==3.3.3
matplotlib==3.9.0
joblib==1.4.2
gunicorn==22.0.0
```

> 💡 *Note: Version numbers may vary depending on your Python environment.*

To install all dependencies:

```bash
pip install -r requirements.txt
```

---

### 🧠 Tech Stack

* **Language:** Python
* **Libraries:** TensorFlow, Keras, NumPy, Pandas, Scikit-learn
* **Web Framework:** Flask
* **Deployment:** Ready for local and cloud deployment

---

### 🏁 Conclusion

This project demonstrates a complete **Deep Learning Regression Pipeline**, from data preprocessing and training to deployment as a **Flask web app**. It showcases how neural networks can effectively predict house prices and serve as a robust model deployment example.

---

Would you like me to make a **ready-to-use `requirements.txt` file** (you can download directly) with the above dependencies?

























[//]: # (# House-Price-Prediction-Using-Deep-learning-ANN)

[//]: # ()
[//]: # (Table of Contents)

[//]: # ()
[//]: # ()
[//]: # (Introduction)

[//]: # (Data Preprocessing)

[//]: # (Neural Network Architecture)

[//]: # (Model Training)

[//]: # (Model Evaluation)

[//]: # (Web Application using Flask)

[//]: # (Documentation for Flask App)

[//]: # (1. Introduction)

[//]: # ()
[//]: # (The House Price Prediction System is designed to predict house prices based on various features like location, room details, and other amenities. The system utilizes a neural network model implemented using the Keras framework. Additionally, a Flask web application has been developed to provide an intuitive interface for users to input data and receive price predictions.)

[//]: # ()
[//]: # ()
[//]: # (2. Data Preprocessing)

[//]: # (Data Scaling: Features are scaled using the Min-Max Scaler from scikit-learn to ensure consistent input to the neural network.)

[//]: # (Data Splitting: The dataset is split into training and testing sets to evaluate the model's performance accurately.)

[//]: # ()
[//]: # (3. Neural Network Architecture)

[//]: # (The neural network architecture consists of an input layer, two hidden layers with dropout for regularization, and an output layer for regression.)

[//]: # ()
[//]: # (Input Layer: 1000 neurons, ReLU activation)

[//]: # (Dropout Layer: 20% dropout rate)

[//]: # (Hidden Layer 1: 500 neurons, ReLU activation)

[//]: # (Dropout Layer: 20% dropout rate)

[//]: # (Hidden Layer 2: 250 neurons, ReLU activation)

[//]: # (Output Layer: 1 neuron, linear activation for regression)

[//]: # (4. Model Training)

[//]: # ()
[//]: # (The model is compiled using the RMSprop optimizer and Mean Squared Error loss function. Early stopping is implemented with a patience of 50 epochs to prevent overfitting. The training is performed for 10 epochs with a batch size of 50.)

[//]: # ()
[//]: # (5. Model Evaluation)

[//]: # ()
[//]: # (The model is evaluated using various metrics such as Mean Absolute Error &#40;MAE&#41;, Mean Squared Error &#40;MSE&#41;, Mean Squared Log Error &#40;MSLE&#41;, and R-squared score on the testing dataset.)

[//]: # ()
[//]: # (6. Web Application using Flask)

[//]: # ()
[//]: # (A Flask web application has been created to provide a user-friendly interface for predicting house prices. The application loads the pre-trained neural network model and scaler. Users can input information such as longitude, latitude, house age, and other details to get a predicted house price.)

[//]: # ()
[//]: # ()
[//]: # (7. How to run th Project)

[//]: # (.venv\Scripts\activate)

[//]: # (python app.py)
