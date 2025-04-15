# Mildew Detection in Cherry Leaves

## Project Overview

### Description
Farmy & Foods, an agricultural company specializing in food production and harvesting, has identified a critical issue with powdery mildew affecting their cherry plantations—one of their flagship products. Powdery mildew is a fungal disease that compromises the quality of cherry leaves and, consequently, the cherries themselves. The current manual inspection process involves employees spending 30 minutes per tree to visually assess leaves for mildew, followed by a 1-minute treatment if mildew is detected. With thousands of cherry trees across multiple farms, this process is time-intensive and unscalable.

To address this challenge, the IT team at Farmy & Foods, led by Marianne McGuineys (Head of IT and Innovation), proposed a Machine Learning (ML) system to instantly detect powdery mildew from cherry leaf images. This project develops an ML-powered dashboard that visually differentiates healthy cherry leaves from those with powdery mildew and predicts the health status of cherry trees. The solution aims to save time, improve scalability, and ensure product quality, with potential applications to other crops in the future.

The dataset used is a collection of cherry leaf images provided by Farmy & Foods, sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).

---

## Table of Contents
1. [Description](#description)
2. [Business Requirements](#business-requirements)
3. [Dataset](#dataset)
4. [User Stories](#user-stories)
5. [ML Business Case](#ml-business-case)
6. [Dashboard Design](#dashboard-design)
7. [Data Collection and Processing](#data-collection-and-processing)
8. [Data Analysis and Visualization](#data-analysis-and-visualization)
9. [Machine Learning Pipeline](#machine-learning-pipeline)
10. [Conclusions](#conclusions)
11. [Libraries and Tools](#libraries-and-tools)

## Business Requirements

1. **Visual Differentiation**: Conduct a study to visually differentiate healthy cherry leaves from those with powdery mildew.
2. **Prediction**: Predict whether a cherry tree is healthy or contains powdery mildew based on leaf images.
3. **Dashboard**: Deliver a user-friendly dashboard to meet the above requirements.

---

## Dataset

The dataset consists of cherry leaf images collected from Farmy & Foods' plantations. It includes:
- Images labeled as "healthy" or "powdery_mildew."
- Split into training, validation, and test sets for ML model development.
- Available at: [Kaggle - Cherry Leaves](https://www.kaggle.com/codeinstitute/cherry-leaves).

---

## User Stories

### User Story 1: Visual Differentiation
- **As a** Farmy & Foods employee,
- **I want** to visually compare healthy and powdery mildew-affected cherry leaves,
- **So that** I can understand the differences and confirm the ML model's reliability.
- **Tasks**: Create an image montage, compute average and variability images, and display differences between healthy and mildew-affected leaves on the dashboard.

### User Story 2: Prediction
- **As a** Farmy & Foods employee,
- **I want** to upload a cherry leaf image and receive an instant prediction of its health status,
- **So that** I can quickly decide if treatment is needed without manual inspection.
- **Tasks**: Develop an ML pipeline to classify leaf images and integrate it into the dashboard for real-time predictions.

---

## Rationale to Map Business Requirements to Data Visualizations and ML Tasks

- **Visual Differentiation**: Data visualizations (image montage, average images, variability images) enable employees to study the visual characteristics of healthy vs. mildew-affected leaves, fulfilling Business Requirement 1.
- **Prediction**: An ML classification task predicts whether a leaf is healthy or contains powdery mildew, addressing Business Requirement 2. The model outputs a binary classification (healthy/mildew) actionable for treatment decisions.

---

## ML Business Case

### Task: Powdery Mildew Detection
- **Aim**: Predict if a cherry leaf is healthy or has powdery mildew using image data.
- **Learning Method**: Supervised learning (binary classification) with a Convolutional Neural Network (CNN).
- **Ideal Outcome**: Instantly identify mildew-affected leaves, reducing inspection time from 30 minutes to seconds per tree.
- **Success Metrics**: Achieve at least 95% accuracy on the test set; minimize false negatives to ensure mildew is not missed.
- **Failure Metrics**: Accuracy below 85% or high false negative rate, risking untreated mildew spread.
- **Model Output**: Binary classification ("Healthy" or "Powdery Mildew") with a confidence score.
- **Relevance for User**: Employees can prioritize treatment, saving time and ensuring product quality.
- **Heuristics**: Image augmentation (rotation, flipping) to enhance model robustness.
- **Training Data**: Cherry leaf images from Farmy & Foods, split into train/validation/test sets.

---

## Dashboard Design

The dashboard is built using Flask and deployed on Heroku, accessible at [Mildew Detection App](https://mildew-detection-ellis-2025.herokuapp.com/). It includes the following routes:

- **Home Page (`/home`)**:
  - **Content**: Introduces the project, outlining the problem of powdery mildew and the ML solution.
  - **Business Requirement**: Provides context for both requirements.

- **Dataset Page (`/dataset`)**:
  - **Content**: Describes the Kaggle dataset, including image counts and train/validation/test splits.
  - **Business Requirement**: Supports visual differentiation by explaining data context.

- **Visual Study Page (`/visual_study`)**:
  - **Content**: Displays image montage, average images for healthy and mildew-affected leaves, variability images, and difference plots.
  - **Business Requirement**: Answers Requirement 1 (visual differentiation).
  - **Interpretation**: Montage shows example leaves; average images highlight typical features (e.g., white patches for mildew); variability shows consistency; difference plot emphasizes mildew-specific patterns.

- **Prediction Page (`/predict`)**:
  - **Content**: File upload form for cherry leaf images, prediction result (Healthy/Powdery Mildew with confidence score), and uploaded image stored in Google Cloud Storage.
  - **Business Requirement**: Answers Requirement 2 (prediction).
  - **Interpretation**: Prediction result informs the user if treatment is needed; storage enables tracking.

- **Performance Page (`/performance`)**:
  - **Content**: Plots of training loss, accuracy, confusion matrix, and precision-recall curves; statement of model performance.
  - **Business Requirement**: Validates both requirements by showing model metrics.
  - **Interpretation**: Metrics confirm model reliability (99.37% test accuracy).

- **Hypothesis Page (`/hypothesis`)**:
  - **Content**: Outlines project hypotheses (e.g., ML can reduce inspection time) and outcomes.
  - **Business Requirement**: Supports project context.

---

## Data Collection and Processing

- Data is collected from the Kaggle endpoint using `notebooks/01_data_collection.ipynb`.
- Images are preprocessed (resized, augmented) and loaded into memory via `notebooks/02_data_preprocessing.ipynb`.

---

## Data Analysis and Visualization

- Analysis in `notebooks/02_data_preprocessing.ipynb` includes:
  - Setting image shape.
  - Computing average and variability images.
  - Difference between average healthy and mildew images.
  - Image montage.
  - Plot of dataset split (train/validation/test).

---

## Machine Learning Pipeline

- Developed in `notebooks/03_model_training.ipynb`:
  - CNN architecture (MildewCNN) with dropout and early stopping to prevent overfitting.
  - Hyperparameter optimization for learning rate and batch size.
  - Feature importance assessed via model performance.
  - Model saved to Google Cloud Storage (`gs://mildew-detection-uploads-2025/mildew_cnn_model_trained.pth`).
- Evaluated in `notebooks/03_model_training.ipynb`:
  - Learning curves (loss/accuracy) for train/validation sets.
  - Confusion matrix and classification report for test set.
  - Achieved 99.37% test accuracy, exceeding the 95% success metric.

---

## Conclusions

- **Data Analytics**: Healthy leaves show uniform green color, while mildew-affected leaves exhibit white powdery patches, as seen in average and difference images.
- **ML Performance**: The model successfully predicts leaf health with 99.37% accuracy, meeting the business requirement for instant detection and reducing manual inspection time.

---

## Libraries and Tools

- Python
- PyTorch
- Flask
- Google Cloud Storage
- Jupyter Notebook
- Kaggle
- GitHub
- Youtube
- Heroku
- Slack
- ChatGpt 
- Pandas
- NumPy
- Matplotlib
- Seaborn
# Updated to ensure sync
