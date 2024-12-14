import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import optuna
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import train_test_split

X, y = load_train()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.replace('?','Unknown',inplace=True)
X_test.replace('?','Unknown',inplace=True)

X_train['workclass'].replace(['Federal-gov','Local-gov','State-gov'],'Gov',inplace=True)
X_train['workclass'].replace(['Self-emp-not-inc','Self-emp-inc'],'Self', inplace=True)
X_train['workclass'].replace(['Never-worked','Without-pay', 'Unknown'],'Other/Unknown ',inplace=True)

X_test['workclass'].replace(['Federal-gov','Local-gov','State-gov'],'Gov', inplace=True)
X_test['workclass'].replace(['Self-emp-not-inc','Self-emp-inc'],'Self', inplace=True)
X_test['workclass'].replace(['Never-worked','Without-pay', 'Unknown'],'Other/Unknown', inplace=True)
# -------------------------------------------------------------------------------------------------------------------
X_train['education'].replace(['Preschool', '1st-4th', '5th-6th', '7th-8th'], 'Elem', inplace=True)
X_train['education'].replace(['9th', '10th', '11th', '12th', 'HS-grad'], 'HS-grad', inplace=True)
X_train['education'].replace(['Assoc-acdm', 'Assoc-voc'], 'Assoc', inplace=True)
# X_train['education'].replace(['Bachelors', 'Masters', 'Doctorate', 'Prof-school'], 'Graduate', inplace=True)
# 'Some-college',
X_test['education'].replace(['Preschool', '1st-4th', '5th-6th', '7th-8th'], 'Elem', inplace=True)
X_test['education'].replace(['9th', '10th', '11th', '12th', 'HS-grad'], 'HS-grad', inplace=True)
X_test['education'].replace(['Assoc-acdm', 'Assoc-voc'], 'Assoc', inplace=True)
# X_test['education'].replace(['Bachelors', 'Masters', 'Doctorate', 'Prof-school'], 'Graduate', inplace=True)
# 'Some-college',
# -------------------------------------------------------------------------------------------------------------------
#  , 'Adm-clerical'
X_train['occupation'].replace(['Adm-clerical', 'Tech-support'], 'White-Collar', inplace=True)
X_train['occupation'].replace(['Craft-repair', 'Machine-op-inspct', 'Farming-fishing'],'Blue-Collar', inplace=True)
X_train['occupation'].replace(['Priv-house-serv', 'Handlers-cleaners','Other-service','Transport-moving',], 'Manual-Labor', inplace=True)
X_train['occupation'].replace(['Prof-specialty','Protective-serv','Exec-managerial'], 'Professional', inplace=True)
X_train['occupation'].replace(['Armed-Forces', 'Unknown'], 'Other/Unknown', inplace=True)

X_test['occupation'].replace(['Adm-clerical', 'Tech-support'], 'White-Collar', inplace=True)
X_test['occupation'].replace(['Craft-repair','Machine-op-inspct', 'Farming-fishing'],'Blue-Collar', inplace=True)
X_test['occupation'].replace(['Priv-house-serv', 'Handlers-cleaners','Other-service','Transport-moving',], 'Manual-Labor', inplace=True)
X_test['occupation'].replace(['Prof-specialty','Protective-serv','Exec-managerial'], 'Professional', inplace=True)
X_test['occupation'].replace(['Armed-Forces', 'Unknown'], 'Other/Unknown', inplace=True)

# # -------------------------------------------------------------------------------------------------------------------

X_train['marital.status'].replace(['Married-civ-spouse','Married-AF-spouse'], 'Married', inplace=True)
X_train['marital.status'].replace(['Divorced', 'Separated','Widowed', 'Married-spouse-absent'], 'Previously-Married', inplace=True)
# 'Never-married', 
X_test['marital.status'].replace(['Married-civ-spouse','Married-AF-spouse','Married-spouse-absent'], 'Married', inplace=True)
X_test['marital.status'].replace(['Divorced', 'Separated','Widowed', 'Married-spouse-absent'], 'Previously-Married', inplace=True)
# 'Never-married', 
# -------------------------------------------------------------------------------------------------------------------

X_train['relationship'].replace(['Not-in-family', 'Unmarried', 'Other-relative'],'Not-in-family', inplace=True)
# X_train['relationship'].replace(['Husband', 'Own-child', 'Wife'], 'Family', inplace=True)
X_test['relationship'].replace(['Not-in-family', 'Unmarried', 'Other-relative'],'Not-in-family', inplace=True)
# X_test['relationship'].replace(['Husband', 'Own-child', 'Wife'], 'Family', inplace=True)
# -------------------------------------------------------------------------------------------------------------------

X_train.loc[X_train['native.country']!=' United-States', 'native.country'] = 'Non-US'
X_train.loc[X_train['native.country'] == ' United-States', 'native.country'] = 'US'
X_test.loc[X_test['native.country']!=' United-States', 'native.country'] = 'Non-US'
X_test.loc[X_test['native.country'] == ' United-States', 'native.country'] = 'US'

X_train, X_test = ecode_onehot(X_train,X_test)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

# Model
def create_model(trial):
    input_dim = X_train.shape[1]
    layers = []
    n_layers = trial.suggest_int("n_layers", 2, 6)
    in_features = input_dim

    # Activation
    activation_functions = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'LeakyReLU': nn.LeakyReLU()
    }
    activation_choice = trial.suggest_categorical("activation_function", list(activation_functions.keys()))

    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 32, 512)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation_functions[activation_choice])
        p = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

# Train and Test
def train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, epochs=30, patience=5):
    best_auc = 0
    best_model_state = None
    train_losses, valid_losses = [], []
    train_aucs, valid_aucs = [], []
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        y_train_true, y_train_scores = [], []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            y_train_true.extend(y_batch.numpy())
            y_train_scores.extend(y_pred.detach().numpy())

        train_auc = roc_auc_score(y_train_true, y_train_scores)
        train_losses.append(total_train_loss / len(train_loader))
        train_aucs.append(train_auc)

        # Validation
        model.eval()
        total_valid_loss = 0
        y_valid_true, y_valid_scores = [], []

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_valid_loss += loss.item()
                y_valid_true.extend(y_batch.numpy())
                y_valid_scores.extend(y_pred.numpy())

        valid_auc = roc_auc_score(y_valid_true, y_valid_scores)
        valid_losses.append(total_valid_loss / len(valid_loader))
        valid_aucs.append(valid_auc)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_losses[-1]:.4f}, Train AUC: {train_auc:.4f} | Valid Loss: {valid_losses[-1]:.4f}, Valid AUC: {valid_auc:.4f}")

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    return train_losses, valid_losses, train_aucs, valid_aucs

# Optuna Objective func
def objective(trial):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    for train_idx, valid_idx in kf.split(X_train):
        X_fold_train, X_fold_valid = X_train[train_idx], X_train[valid_idx]
        y_fold_train, y_fold_valid = y_train[train_idx], y_train[valid_idx]

        train_loader = DataLoader(TensorDataset(X_fold_train, y_fold_train), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(X_fold_valid, y_fold_valid), batch_size=batch_size, shuffle=False)

        model = create_model(trial)
        criterion = nn.BCELoss()

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        auc = train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, epochs=30, patience=5)[-1][-1]
        auc_scores.append(auc)

    return np.mean(auc_scores)

# Optuna Run
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Optimal hyperparameters
print("Best Hyperparameters:", study.best_params)

# Optimal model train and Test
best_model = create_model(study.best_trial)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=study.best_params['learning_rate'])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=study.best_params['batch_size'], shuffle=True)
valid_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=study.best_params['batch_size'], shuffle=False)

train_losses, valid_losses, train_aucs, valid_aucs = train_and_evaluate(best_model, train_loader, valid_loader, criterion, optimizer)
print(f"Final Test AUC: {roc_auc_score(y_test.numpy(), best_model(X_test).detach().numpy()):.4f}")