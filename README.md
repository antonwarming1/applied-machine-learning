# Applied Machine Learning: Steel Plate Fault Classification

A mini project demonstrating binary and multiclass classification techniques on steel plate fault detection using machine learning algorithms.

## Overview

This project applies machine learning techniques to classify steel plate faults using the Steel Plates Faults dataset from the UCI Machine Learning Repository. The project includes two main approaches:

1. **Binary Classification** (`binary_model_applied.py`): Classifies faults into minor faults (Stains, Dirtiness, Pastry) vs. major faults(Z_Scratch, K_Scatch, Bumps, Other_Faults) Decision Trees and Random Forest with hyperparameter tuning.

2. **Multiclass Classification** (`multiclass_apllied.py`): Classifies faults into specific categories (Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults) using neural networks implemented in PyTorch.

## Dataset

The dataset used is the https://archive.ics.uci.edu/dataset/198/steel+plates+faults. It contains 27 features describing steel plate characteristics and 7 fault types.

**Features include:**
- X_Minimum
- X_Maximum
- Y_Minimum
- Y_Maximum
- Pixels_Areas
- X_Perimeter
- Y_Perimeter
- Sum_of_Luminosity
- Minimum_of_Luminosity
- Maximum_of_Luminosity
- Length_of_Conveyer
- TypeOfSteel_A300
- TypeOfSteel_A400
- Steel_Plate_Thickness
- Edges_Index
- Empty_Index
- Square_Index
- Outside_X_Index
- Edges_X_Index
- Edges_Y_Index
- Outside_Global_Index
- LogOfAreas
- Log_X_Index
- Log_Y_Index
- Orientation_Index
- Luminosity_Index
- SigmoidOfAreas

**Target variables:**
- Pastry
- Z_Scratch
- K_Scatch
- Stains
- Dirtiness
- Bumps
- Other_Faults


## The classification problems
## Binary Classification Approach

### Models Used
- Decision Tree Classifier
- Random Forest Classifier

### Techniques Applied
- Hyperparameter tuning using Grid Search CV and Randomized Search CV and Baysian optimization to find best hyperparameters
- Stratified K-Fold cross-validation (due to imbalanced dataset)


### Key Features
- Handles imbalanced dataset with stratification
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Confusion matrix analysis
- Feature importance analysis

## Multiclass Classification Approach

### Model Used
- Neural Network implemented in PyTorch

### Techniques Applied
- Label encoding for multiclass targets
- Standard scaling of features
- Optuna for hyperparameter optimization

### Key Features
- Custom neural network architecture
- Handles imbalanced dataset with stratification
- Training with validation monitoring
- Performance evaluation on test set
- Confusion matrix visualization

## Requirements for both

python 3.12.9

install these:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
optuna
torch
torchvision
requests
urllib3
scipy

### Time

As the gridsearch is done for 3 different algorimts in binary classification and for 1 algoritme in multi class, the code does take some time to run. For the multiclass if an nvidia GPU is avilible it can be used to make it alot faster.
