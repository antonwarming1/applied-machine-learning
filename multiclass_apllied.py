import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import requests
from io import StringIO
import urllib3
import optuna
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import copy
from sklearn.preprocessing import LabelEncoder



# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Download the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"
response = requests.get(url, verify=False)  # Bypass SSL verification
data = StringIO(response.text)

# Read the data into a pandas DataFrame
df = pd.read_csv(data, sep=r"\s+", header=None)
print(df.head())
print(f"data shape: {df.shape}")
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



fault_types=['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']


# Check for missing values
print(f'missing values:\n{df.isnull().sum()}')
# no missing values found

# Create target column (get the fault type name for each row)
df['target'] = df[fault_types].idxmax(axis=1)

# Drop the individual fault columns
df = df.drop(columns=fault_types)



# Data Exploration
#print(df.head(10))
#print(f'info:{df.info()}')
#print(df.describe())


# Visualize the distribution of fault types
target_counts=[]
for target in df['target'].unique():
    count=df[df['target']==target].shape[0]
    target_counts.append((target,count))

#plot the distribution
plt.figure(figsize=(10, 7))
sns.countplot(x='target', data=df)
plt.title('Multiclass Fault Distribution')
plt.xticks(rotation=45)
for i, (target, count) in enumerate(target_counts):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)
plt.show()

# Prepare data for PyTorch model training
X = df.drop('target', axis=1)
y = df['target']
print(y.shape)

# Encode string labels to integers (0-6) for CrossEntropyLoss

le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Show mapping of classes
print(f"Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Simple train/validation split (80% train, 20% validation) where stratify to maintain class distribution
x_train, x_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=32, stratify=y_encoded)
# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# Convert to PyTorch tensors
x_train = torch.from_numpy(x_train).type(torch.float)
x_val = torch.from_numpy(x_val).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.long)
y_val = torch.from_numpy(y_val).type(torch.long)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# Model parameters global
input_features = x_train.shape[1]  # 27 features
output_classes = 7  # 7 fault types

# Define PyTorch neural network model
class FaultsNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation_type='relu'):
        super().__init__()
        layers = []
        in_features = input_dim
        
        # Build hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            
            # Add activation function
            if activation_type == 'relu':
                layers.append(nn.ReLU())
            elif activation_type == 'tanh':
                layers.append(nn.Tanh())
            elif activation_type == 'sigmoid':
                layers.append(nn.Sigmoid())
            #first layer wth input features size
            in_features = hidden_dim
        
        # Output layer (no activation - will use softmax in loss function)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    # Define forward pass
    def forward(self, x):
        return self.network(x)

# Define model dimensions for standard model
def standard_model():
    # Define model dimensions
    hidden_features = 550
    num_hidden_layers = 2


    # setup model_0
    model_0 = FaultsNN(input_dim=input_features, 
                    hidden_dim=hidden_features, 
                    output_dim=output_classes, 
                    num_layers=num_hidden_layers,
                    activation_type='relu').to(device)
    return model_0


def train_model(model, x_train, y_train, x_val, y_val, loss_fn, optimizer, epochs=1200, patience=10):
    #Train model with early stopping based on validation loss
    torch.manual_seed(42)
    model.train()
    #set initial best validation loss to infinity 
    best_val_loss = float('inf')
    # Initialize counter for epochs with no improvement
    epochs_no_improve = 0
    # Store the best model state
    best_model_state = None
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        y_pred = model(x_train)
        train_loss = loss_fn(y_pred, y_train)
        #
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val)
            val_pred = torch.softmax(val_logits, dim=1).argmax(dim=1)
            val_loss = loss_fn(val_logits, y_val)
            val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
        
        # Early stopping check and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model state
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        #print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
        
        # Stop if no improvement for 'patience' epochs
        if epochs_no_improve >= patience:
            # Load the best model state before stopping
            model.load_state_dict(best_model_state)
            break
    
    return model

def objective(trial):
    #Optuna objective function for PyTorch neural network hyperparameter optimization
    
    # Select hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
    num_layers = trial.suggest_int("num_layers", 2, 5)
    activation_type = trial.suggest_categorical("activation_type", ["relu", "tanh"])
    lr = trial.suggest_float("lr", 0.01, 0.05, log=True)
    
    # build model
    model = FaultsNN(
        input_dim=input_features,
        hidden_dim=hidden_dim,
        output_dim=output_classes,
        num_layers=num_layers,
        activation_type=activation_type
    ).to(device)
    # setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # train model
    trained_model = train_model(
        model,
        x_train.to(device), y_train.to(device),
        x_val.to(device), y_val.to(device),
        loss_fn, optimizer,
        epochs=1000,     
        patience=15
    )

    # evaluate model on validation set
    trained_model.eval()
    with torch.inference_mode():
        logits = trained_model(x_val.to(device))
        preds = torch.argmax(logits, dim=1)
    # Move to CPU for sklearn metrics
    y_true = y_val.cpu().numpy()
    y_pred = preds.cpu().numpy()
    # Calculate accuracy
    val_acc = accuracy_score(y_true, y_pred)
    
    # We want to maximize accuracy, so we return it directly
    return val_acc

def run_optuna_optimization(n_trials=100):
    # Run Optuna optimization for PyTorch neural network hyperparameters
    
    print(f"\n=== RUNNING OPTUNA OPTIMIZATION ({n_trials} trials) ===")
    # Create study and optimize and set random seed for reproducibility
    sampler =  optuna.samplers.TPESampler(seed=10) 
    study = optuna.create_study(direction='maximize', study_name='pytorch_nn_optimization', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters
    best_params = study.best_params
    loss_fn = nn.CrossEntropyLoss()
    # Build final model
    final_model = FaultsNN(
        input_dim=input_features,
        hidden_dim=best_params['hidden_dim'],
        output_dim=output_classes,
        num_layers=best_params['num_layers'],
        activation_type=best_params['activation_type']
    ).to(device)
    # Setup optimizer for final model and loss function
    final_optimizer = torch.optim.SGD(final_model.parameters(), lr=best_params['lr'], momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    print("\n=== TRAINING FINAL MODEL ===")
    final_model = train_model(
        final_model,
        x_train.to(device),
        y_train.to(device),
        x_val.to(device),
        y_val.to(device),
        loss_fn,
        final_optimizer,
        epochs=500,
        patience=15
    )
    
    return final_model, study.best_params

def evaluate_model(model, x_test, y_test, class_names=None):
    """Comprehensive evaluation of PyTorch model"""
    model.eval()
    with torch.inference_mode():
        # Get predictions
        y_pred = model(x_test)
        # Get predicted classes with argmax
        y_pred_classes = torch.argmax(y_pred, dim=1)
    
    # Move to CPU for sklearn metrics
    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_pred_classes.cpu().numpy()
    
    # Calculate metrics and print them
    accuracy = accuracy_score(y_test_np, y_pred_np)
    precision, recall, f1, support = precision_recall_fscore_support(y_test_np, y_pred_np)
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"Class {class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-Score: {f1[i]:.4f}")
        print(f"  Support: {support[i]}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    # setup confusion matrix
    cm = confusion_matrix(y_test_np, y_pred_np)
    cm_normalized= confusion_matrix(y_test_np, y_pred_np, normalize='true')
    
    
    
    
    return accuracy, cm, cm_normalized
         
    

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    
    #Plot confusion matrix.
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    # Run Optuna optimization
    best_model, best_params = run_optuna_optimization(n_trials=100)
    
    # Evaluate on validation set
    print("\n=== EVALUATING ON VALIDATION SET ===")
    # Get class names from label encoder
    class_names = le.classes_
    # Evaluate best model
    accuracy, cm, cm_normalized = evaluate_model(best_model, x_val.to(device), y_val.to(device), class_names=class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names=class_names, title=f' Optuna Optimized (Val Acc: {accuracy})')
    3# Plot normalized confusion matrix
    plot_confusion_matrix(cm_normalized, class_names=class_names, title=f'Optuna Optimized Normalized (Val Acc: {accuracy})')
    
    print("\n=== FINAL BEST HYPERPARAMETERS ===")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    # Train and evaluate a standard model for comparison
    model_0 = standard_model()
    # Setup loss function 
    loss_fn = nn.CrossEntropyLoss()
    # Setup optimizer with learning rate fine-tuned for model_0
    optimizer = torch.optim.SGD(model_0.parameters(),lr=0.022, momentum=0.9)
    # Train model_0
    trained_model = train_model(model_0, 
                                x_train.to(device), y_train.to(device), 
                                x_val.to(device), y_val.to(device), 
                                loss_fn, optimizer, epochs=1000, patience=15)
    print(f"conting y_val classes each class:{torch.bincount(y_val)}" )
    # Evaluate base model
    accuracy_base, cm_base, cm_normalized_base = evaluate_model(trained_model, x_val.to(device), y_val.to(device), class_names=class_names)
    # Plot confusion matrix for base model
    plot_confusion_matrix(cm_base, class_names=class_names,title=f'Base Model Confusion Matrix (Val acc: {accuracy_base})')
    #plot normalized confusion matrix for base model
    plot_confusion_matrix(cm_normalized_base, class_names=class_names, title=f'Base Model Normalized Confusion Matrix(Val acc: {accuracy_base})')
    # Side-by-side comparison plot of base model vs optuna optimized model
    plt.figure(figsize=(8, 5))
    models = ['Base NN', 'Optuna Optimized NN']
    scores = [accuracy_base, accuracy]
    sns.barplot(x=models, y=scores)
    plt.ylim(0, 1)
    plt.ylabel('Validation Set Accuracy')
    plt.title('Base NN vs Optuna Optimized NN - Validation Set Performance')
    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=12)
    plt.show()
    
    
    
    
        
if __name__ == "__main__":
    
    main()
    
    
    
    