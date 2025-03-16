# Predicting Student Performance using Machine Learning

## Overview
This project applies machine learning techniques to predict student performance based on historical academic records. It implements probabilistic matrix factorization (PMF), ensemble learning models, and feature engineering techniques to enhance prediction accuracy.

## Project Structure
```
Predicting-Student-Performance/
│── src/                          # Source Code
│   │── module1.py                 # Correlation Analysis
│   │── module2.py                 # Grade Correlation & Course Selection
│   │── module3.py                 # Matrix Factorization Model
│   │── module4.py                 # Predictive Model
│   │── module5.py                 # Performance Prediction
│   │── pmf.py                      # Probabilistic Matrix Factorization
│── data/                         # Datasets
│   │── U.xlsx                      # Student Performance Data
│   │── V.xlsx                      # Course Mapping Data
│   │── datasets.csv                 # Additional Dataset
│── docs/                         # Documentation
│   │── seminarfinal.pptx           # Presentation
│   │── 121003079_Abstract.pdf      # Research Abstract
│── README.md                     # Project Overview
│── requirements.txt               # Dependencies
│── .gitignore                     # Ignoring large dataset files
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-link>
   cd Predicting-Student-Performance
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main prediction model:
   ```bash
   python src/module5.py
   ```

## Dependencies
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Machine Learning Models Used
- **Decision Trees**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest Regressor**
- **Probabilistic Matrix Factorization (PMF)**

## Probabilistic Matrix Factorization (PMF)
PMF is used to fill missing values in the dataset by factorizing the student-course interaction matrix into latent student and subject features. The optimization equations used for updating the matrices U (students) and V (subjects) are:

<img width="350" alt="image" src="https://github.com/user-attachments/assets/33510b95-51fd-4eb9-9b24-74dec46d8624" />


The model is optimized using log-a posteriori maximization:

<img width="608" alt="image" src="https://github.com/user-attachments/assets/ded8e7f6-cdeb-4ec7-875f-6fc3dd9ef12b" />

These equations enable the model to estimate missing student grades by learning from existing course interactions.


## Results
The model evaluates student performance using mean squared error (MSE) and prediction accuracy based on historical course data. It uses a bi-layered approach for progressive learning and prediction refinement.

## References
- Research on Student Performance Prediction Models
- Data sourced from UCLA academic records

---
Developed as part of an academic project to explore predictive analytics in education.

