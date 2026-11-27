# Fashion-MNIST Image Classification

## Project Overview
This project focuses on classifying fashion images from the Fashion-MNIST dataset using Support Vector Machine (SVM) algorithms. It compares two approaches: Raw-Pixel SVM and HOG-based SVM.

## Phase 1 – HOG-SVM Classification
- Preprocessing of Fashion-MNIST images
- Histogram of Oriented Gradients (HOG) feature extraction
- SVM model training and evaluation
- Saving the trained model

## Phase 2 – Model Comparison and Application
- Comparison of Raw-Pixel SVM and HOG-SVM
- Evaluation using accuracy, precision, recall and F1-score
- Interactive image classification using Streamlit
- Experiment tracking using MLflow
- Docker-based application deployment

## Technologies Used
- Python
- NumPy
- Scikit-learn
- OpenCV
- Streamlit
- MLflow
- Docker

## Models
- Raw-Pixel SVM
- HOG-SVM

## How to Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Objective
To study the effect of feature extraction on image classification performance by comparing raw pixel features with HOG features using SVM.