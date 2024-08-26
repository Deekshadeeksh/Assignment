import os
from src.functions import load_and_clean_data, perform_eda, feature_engineering, train_and_evaluate_models, save_model
from sklearn.model_selection import train_test_split

def main():

    data = load_and_clean_data('data/game_user_churn.csv')


    perform_eda(data)


    data = feature_engineering(data)


    X = data.drop(columns=['churn'])
    y = data['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    best_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)


    best_model_name = 'Random Forest'
    save_model(best_models[best_model_name], 'src/models/best_model.pkl')


if __name__ == "__main__":
    main()

