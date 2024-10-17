from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json

# Load dataset
DATA_PATH = 'C:\\Users\\DENI\\Desktop\\pr\\udemy_courses.csv'  # Adjust to your path

# Data preprocessing
def preprocess_data(data):
    data['published_timestamp'] = pd.to_datetime(data['published_timestamp'], errors='coerce')
    data['Year'] = data['published_timestamp'].dt.year
    data['Month'] = data['published_timestamp'].dt.month
    return data

# Feature engineering
def feature_engineering(data):
    label_encoder = LabelEncoder()
    data['subject'] = label_encoder.fit_transform(data['subject'])
    data['level'] = label_encoder.fit_transform(data['level'])
    
    features = ['subject', 'level', 'num_lectures', 'content_duration', 'price', 'Year', 'Month']
    X = data[features].copy()
    y = (data['num_subscribers'] == data['num_subscribers'].max()).astype(int)

    # Normalize numerical features
    scaler = StandardScaler()
    X[['num_lectures', 'content_duration', 'price']] = scaler.fit_transform(X[['num_lectures', 'content_duration', 'price']])

    return X, y

def search_view(request):
    if request.method == "POST":
        try:
            # Load and preprocess the dataset
            data = pd.read_csv(DATA_PATH)
            data = preprocess_data(data)

            data_request = json.loads(request.body)
            year = int(data_request.get('year'))
            month = int(data_request.get('month'))

            # Validate input
            if year < 2010 or month not in range(1, 13):
                return JsonResponse({"error": "Invalid year or month values."}, status=400)

            # Filter dataset
            filtered_data = data[(data['Year'] == year) & (data['Month'] == month)].copy()

            if filtered_data.empty:
                return JsonResponse({"top_selling_courses": []})  # No results found

            # Prepare features for prediction
            features = filtered_data[['subject', 'level', 'num_lectures', 'content_duration', 'price', 'Year', 'Month']].copy()
            features.loc[:, 'subject'] = LabelEncoder().fit_transform(features['subject'])
            features.loc[:, 'level'] = LabelEncoder().fit_transform(features['level'])

            # Normalize numerical features
            features[['num_lectures', 'content_duration', 'price']] = StandardScaler().fit_transform(features[['num_lectures', 'content_duration', 'price']])

            # Train model
            X, y = feature_engineering(filtered_data)
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            # Predict top-selling courses
            predictions = model.predict(features)

            # Collect top-selling courses
            filtered_data.loc[:, 'TopSelling'] = predictions
            top_selling_courses = filtered_data[filtered_data['TopSelling'] == 1]

            if top_selling_courses.empty:
                return JsonResponse({"top_selling_courses": []})  # Return empty if no top-selling courses

            courses_result = top_selling_courses[['course_title', 'num_subscribers', 'price', 'url']].to_dict(orient='records')

            return JsonResponse({"top_selling_courses": courses_result}, safe=False)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, 'search/search.html')
