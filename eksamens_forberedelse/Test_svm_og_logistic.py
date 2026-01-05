import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import requests
from io import StringIO
import urllib3
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from binary_model_applied import plot_confusion_matrix, plot_confusion_matrix_normalized
# Download the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"
response = requests.get(url, verify=False)  # Bypass SSL verification
data = StringIO(response.text)

# Read the data into a pandas DataFrame
df = pd.read_csv(data, sep=r"\s+", header=None)

# Assign column names
column_names = [
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 
    'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 
    'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 
    'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 
    'SigmoidOfAreas', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 
    'Dirtiness', 'Bumps', 'Other_Faults'
]

df.columns = column_names


# Define fault categories
surface_faults = ['Stains', 'Dirtiness',"Pastry"]
other_faults = ['Z_Scratch', 'K_Scatch', 'Bumps', 'Other_Faults']



print(f"\nTotal Surface Faults (Stains + Dirtiness): {df[surface_faults].sum().sum()}")
print(f"Total Other Faults: {df[other_faults].sum().sum()}")

# Binary target: 1 = Stains/Dirtiness, 0 = Other faults
df['target'] = np.where(df[surface_faults].any(axis=1), 1, 0)

# Drop the individual fault columns
df = df.drop(columns=surface_faults + other_faults)

# Data Exploration
print(df.head(10))
#print(df.info())
#print(df.describe())

# Check for missing values
print(df.isnull().sum())
#no missing values found

# Visualize the distribution
plt.figure(figsize=(10, 5))

# 1) Fix the order of the bars explicitly
order = [0, 1]   # or sorted(df['target'].unique())
sns.countplot(x='target', data=df, order=order)
plt.title('Binary Fault Distribution (1=Stains, Dirtiness and Pastry, 0=Other)')

# 2) Get counts in the SAME order
counts = df['target'].value_counts().reindex(order)

for i, target in enumerate(order):
    count = counts[target]
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)

plt.show()

print("\nClass Distribution:")
print(df['target'].value_counts())

# Prepare features and target variable
X = df.drop('target', axis=1)
y = df['target']

# duo to very imbalanced dataset, we use stratify

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

def train_logistic_regression(x_train, y_train, x_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy}")
    return model, y_pred

def train_svm(x_train, y_train, x_test, y_test):
    model = SVC(kernel='linear', random_state=42,C=2.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy}")
    return model, y_pred

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    cm = confusion_matrix(y_test, y_pred)
    return cm

def main():
    print("=== LOGISTIC REGRESSION ===")
    # Train Logistic Regression
    lr_model, lr_y_pred = train_logistic_regression(x_train, y_train, x_test, y_test)

    # Evaluate Logistic Regression
    lr_cm = evaluate_model(lr_model, x_test, y_test)
    print("Logistic Regression Confusion Matrix:")
    print(lr_cm)

    print("\n=== SVM ===")
    # Train SVM
    svm_model, svm_y_pred = train_svm(x_train, y_train, x_test, y_test)

    # Evaluate SVM
    svm_cm = evaluate_model(svm_model, x_test, y_test)
    print("SVM Confusion Matrix:")
    print(svm_cm)

    class_names = ['Other Faults', 'Stains/Dirtiness/Pastry']

    # Plot confusion matrices for Logistic Regression
    plot_confusion_matrix(y_test, lr_y_pred, title='Logistic Regression Confusion Matrix')
    plot_confusion_matrix_normalized(y_test, lr_y_pred, title='Logistic Regression Confusion Matrix Normalized')

    # Plot confusion matrices for SVM
    plot_confusion_matrix(y_test, svm_y_pred, title='SVM Confusion Matrix')
    plot_confusion_matrix_normalized(y_test, svm_y_pred, title='SVM Confusion Matrix Normalized')

if __name__ == "__main__":
    main()
