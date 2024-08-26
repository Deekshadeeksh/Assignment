import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


def load_and_clean_data(filepath):
    
    data = pd.read_csv(filepath)


    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col].fillna(data[col].median(), inplace=True)


    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)


    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])

    return data


def perform_eda(data):

    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


    sns.countplot(x='churn', data=data)
    plt.title('Target Variable Distribution')
    plt.show()


def feature_engineering(data):



    if 'total_play_time' in data.columns and 'total_spent' in data.columns:
        data['play_time_per_dollar'] = data['total_play_time'] / (data['total_spent'] + 1)

    if 'age' in data.columns and 'churn' in data.columns:
        data['age_churn_interaction'] = data['age'] * data['churn']



    return data


def train_and_evaluate_models(X_train, X_test, y_train, y_test):

    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])


    param_grid_lr = {
        'model__C': [0.01, 0.1, 1, 10]
    }


    grid_search_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_lr.fit(X_train, y_train)
    best_lr_model = grid_search_lr.best_estimator_


    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])


    param_grid_rf = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30]
    }


    grid_search_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_


    y_pred_lr = best_lr_model.predict(X_test)
    y_pred_rf = best_rf_model.predict(X_test)

    print("Logistic Regression Performance:")
    print(classification_report(y_test, y_pred_lr))
    print("ROC AUC:", roc_auc_score(y_test, best_lr_model.predict_proba(X_test)[:, 1]))

    print("Random Forest Performance:")
    print(classification_report(y_test, y_pred_rf))
    print("ROC AUC:", roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1]))


    return {
        'Logistic Regression': best_lr_model,
        'Random Forest': best_rf_model
    }


def save_model(model, filename):
    joblib.dump(model, filename)

