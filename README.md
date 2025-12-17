### dataset:
Time-series signal features extracted from human motion
These columns are NOT raw sensor readings. They are statistical features extracted from signals, for example:
mean() → average signal value
std() → how much signal varies
energy() → signal power
entropy() → randomness
correlation() → relationship between axes

Example: tBodyAcc-mean()-X
Means:
t → time domain
BodyAcc → body acceleration signal
mean() → average
X → X-axis
So this is: Average value of the signal along X direction

#### Problem Statement -- Human Activity Recognition (HAR) aims to classify physical activities using smartphone sensor data.

#### Dataset

Source: Smartphone accelerometer & gyroscope
Train samples: 7,352
Test samples: 2,947
Features: 561 sensor-based features
Classes: 6 Activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying)

#### Feature engineering categories

Label encoding (target)
Subject-aware separation
Scaling / normalization
Feature grouping (time vs frequency)
Statistical aggregation (energy groups)
Correlation-based pruning
Variance thresholding
Dimensionality reduction (PCA – optional, controlled)

Stage	||   Result
Raw features	|| 561
Engineered	|| 574
After variance filter ||	570
After correlation pruning	|| 253
Final selected	|| 150 


# Tree ensembles dominate HAR due to:
Non-linear sensor patterns
High feature interactions

# Extra Trees outperforms Random Forest because:
More randomness → lower variance
Better generalization on unseen subjects
Handle high-dimensional correlated sensor data
Capture non-linear temporal–frequency interactions
Are robust to noise from wearable sensors
Do not overfit like deep trees in Random Forest


#### Preprocessing
Removed subject column
Label encoded activity classes
No missing values
Features already normalized by dataset creators

#### Model Used
Random Forest Classifier
Handles high-dimensional data well
Robust to noise
Provides feature importance

#### Results

Accuracy: 92.9%
Strong performance across all activities
Best performance on LAYING
Slight confusion between SITTING & STANDING



