#### Human Activity Recognition (HAR) System

### Project Overview

Human Activity Recognition (HAR) aims to classify physical activities performed by a person using smartphone sensor data. This project focuses on sensor-based HAR using time-series features extracted from accelerometer and gyroscope signals.
Unlike raw signal-based approaches, this dataset contains statistical features derived from sensor signals, making it suitable for classical machine learning models.

### Problem Statement

The goal of this project is to accurately classify six human activities using preprocessed smartphone sensor data while ensuring:

- Robust generalization across subjects
- High performance on high-dimensional feature space
- Resistance to sensor noise and feature correlation

### Dataset Description

🔹 Source

- Smartphone accelerometer and gyroscope
- Signals collected from wearable sensors

🔹 Dataset Size
Training || 7,352
Testing || 2,947

🔹 Target Classes (6 Activities)

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

### Feature Description

🔹 Feature Count

- Total features: 561
- These are NOT raw sensor readings
- Features are statistical summaries extracted from time-series signals

🔹 Feature Types
Each feature name encodes detailed information:

Example Feature: tBodyAcc-mean()-X
t- Time-domain signal
BodyAcc - Body acceleration
mean() - Average signal value
X - X-axis

- Interpretation: Average body acceleration along the X-axis in the time domain

### Feature Engineering Categories

The following feature engineering techniques were applied:

- Label encoding of activity classes
- Subject-aware separation (subject column removed)
- Scaling / normalization
- Feature grouping (time-domain vs frequency-domain)
- Statistical aggregation (energy-based groups)
- Correlation-based pruning
- Variance thresholding
- Dimensionality reduction (PCA – optional & controlled)

### Feature Reduction Pipeline

Stage -- Number of Features
Raw features: 561
After engineering: 574
After variance filtering: 570
After correlation pruning: 253
Final selected features: 150

### Preprocessing Steps

- Removed subject identifier
- Encoded activity labels numerically
- No missing values in the dataset
- Features were already normalized by dataset creators
- Applied feature selection to reduce redundancy and multicollinearity

### Model Selection Rationale

🔹 Why Tree-Based Models?
Tree ensemble models dominate HAR tasks due to:

- Strong handling of non-linear sensor patterns
- Ability to model complex feature interactions
- Robustness to noise in wearable sensor data
- Effective learning from high-dimensional feature spaces

🔹 Model Used

- Random Forest Classifier
- Extra Trees evaluated and showed strong performance due to increased randomness and reduced variance.

### Results

🔹 Overall Performance
Accuracy: 92.9%

🔹 Observations

- Strong performance across all activity classes
- Best classification accuracy for LAYING
- Minor confusion observed between: SITTING & STANDING
  This confusion is expected due to similar postural sensor patterns.

### Key Strengths of the System

- High generalization across unseen subjects
- Robust to correlated and redundant features
- Efficient feature reduction without significant performance loss
- Well-suited for real-world wearable sensor applications

### Conclusion

This project demonstrates that classical machine learning models, when combined with careful feature engineering and selection, can achieve high accuracy in Human Activity Recognition tasks without requiring deep learning architectures.
The system is reliable, interpretable, and suitable for deployment in resource-constrained environments.
