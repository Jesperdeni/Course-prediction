import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import logging
import traceback

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Load the dataset
try:
    data = pd.read_csv('C:\\Users\\DENI\\Desktop\\pr\\udemy_courses.csv')
except Exception as e:
    logging.error("Error loading the dataset: %s", e)
    raise

# Print the columns of the DataFrame to check for 'course_title'
logging.info("Columns in the DataFrame: %s", data.columns.tolist())

# Convert 'published_timestamp' to datetime and extract useful time features
data['published_timestamp'] = pd.to_datetime(data['published_timestamp'], errors='coerce')
data['Year'] = data['published_timestamp'].dt.year
data['Month'] = data['published_timestamp'].dt.month

# Initialize Flask app
app = Flask(__name__)

# Check if 'course_title' column exists before proceeding
if 'course_title' in data.columns:
    # Identify repeated course titles
    repeated_titles = data[data.duplicated(['course_title'], keep=False)].copy()

    # Create a binary target variable ('TopSelling') using .loc to avoid warnings
    repeated_titles.loc[:, 'TopSelling'] = repeated_titles.groupby(['Year', 'Month'])['num_subscribers'].transform(lambda x: x == x.max()).astype(int)

    # Feature engineering
    # Encode categorical variables
    label_encoder = LabelEncoder()
    repeated_titles.loc[:, 'subject'] = label_encoder.fit_transform(repeated_titles['subject'])
    repeated_titles.loc[:, 'level'] = label_encoder.fit_transform(repeated_titles['level'])

    # Select features and target variable
    feature_names = ['subject', 'level', 'num_lectures', 'content_duration', 'Year', 'Month']
    X = repeated_titles[feature_names]
    y = repeated_titles['TopSelling']

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Support Vector Machine": SVC()
    }

    # Prepare results for JSON output
    results = {
        "model_performance": {},
        "predictions": {},
        "top_selling_courses": []
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results["model_performance"][model_name] = {
            "accuracy": float(accuracy),  # Convert accuracy to float
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)  # Add classification report as a dictionary
        }

    # Predicting new courses endpoint
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()

            # Expecting data in the format of {"subject": <int>, "level": <int>, "num_lectures": <int>, "content_duration": <float>, "Year": <int>, "Month": <int>}
            new_course = pd.DataFrame(data, index=[0])

            # Predict the new course using each model
            predictions = {}
            for model_name, model in models.items():
                top_selling_prediction = model.predict(new_course)
                predictions[model_name] = int(top_selling_prediction[0])  # Convert to int

            return jsonify(predictions)
        except Exception as e:
            logging.error("Error in predict endpoint: %s", e)
            return jsonify({"error": str(e)}), 400

    # Endpoint to get top-selling courses
    @app.route('/top-selling', methods=['GET'])
    def get_top_selling_courses():
        try:
            # Identify and return the top-selling course titles
            top_selling_courses = repeated_titles[repeated_titles['TopSelling'] == 1]
            result = []
            for index, row in top_selling_courses.iterrows():
                result.append({
                    "course_title": row['course_title'],
                    "month": int(row['Month']),  # Convert to int
                    "year": int(row['Year']),    # Convert to int
                    "subscribers": int(row['num_subscribers'])  # Convert to int
                })
            return jsonify(result)
        except Exception as e:
            logging.error("Error in top-selling endpoint: %s", e)
            return jsonify({"error": str(e)}), 400

else:
    logging.error("The 'course_title' column does not exist in the DataFrame.")
    raise ValueError("The 'course_title' column does not exist in the DataFrame.")

# Run the Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True)  # Change to app.run() for production
    except SystemExit as e:
        logging.error("Server stopped with SystemExit: %s", e)
        traceback.print_exc()
