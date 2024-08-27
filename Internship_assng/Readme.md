Project Overview:
This project is designed to predict user churn in a gaming environment. The aim is to analyze user data to identify patterns and predict whether a user will churn, allowing businesses to implement strategies to retain their users

Project Structure:
The repository is organized as follows:
├── data/
│   └── game_user_churn.csv    # Dataset containing user data
├── src/
│   ├── functions.py           # Functions for data processing, EDA, modeling, and saving models
│   ├── models/
│   │   └── best_model.pkl     # The best-performing model saved after evaluation
├── main.py                    # Main script to run the project workflow
└── requirements.txt           # Python libraries required to run the project


Files and Directories:
data/: Contains the dataset (game_user_churn.csv). Place your dataset here.
src/: Contains the core functions in functions.py and the saved model in the models/ directory.
main.py: The main script that integrates all steps and runs the project workflow.
requirements.txt: Lists the Python libraries needed to run the project.

Steps to Follow:
1. Clone the Repository
Clone this repository to your local machine
2. Install Dependencies
Install the required Python libraries using the following command
3. Add Your Dataset
Ensure your dataset (game_user_churn.csv) is placed in the data/ directory.
4. Running the Project
Run the main.py script to execute the workflow:
This script will:
Load and clean the data: Handle missing values and encode categorical variables.
Perform Exploratory Data Analysis (EDA): Visualize correlations and distributions in the data.
Feature Engineering: Create additional features to improve model performance.
Train and Evaluate Models: Train Logistic Regression and Random Forest models, evaluate their performance, and select the best model.
Save the Best Model: The best model is saved in the src/models/ directory as best_model.pkl.

Model Evaluation:
After running the project, the performance of the Logistic Regression and Random Forest models is displayed. Metrics include:
Precision
Recall
F1-score
ROC AUC Score
The best-performing model is saved for future use.




