import json
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_data(goal):
    """Load and prepare test data for a specific goal."""
    data_file = f'data/training_{goal.lower().replace(" ", "_")}.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Split into train and test (80-20 split)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Prepare features and labels
    X_train = [item['resume_text'] for item in train_data]
    y_train = [item['label'] for item in train_data]
    X_test = [item['resume_text'] for item in test_data]
    y_test = [item['label'] for item in test_data]
    
    return X_train, y_train, X_test, y_test

def evaluate_goal(goal):
    print(f"\nEvaluating model for: {goal}")
    print("=" * 50)
    
    # Load and prepare data
    X_train, y_train, X_test, y_test = load_data(goal)
    
    # Show data stats
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Match', 'Match'])
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Show sample predictions
    print("\nSample Predictions:")
    print("-" * 50)
    for i in range(min(3, len(X_test))):
        actual = "Match" if y_test[i] == 1 else "Not Match"
        pred = "Match" if y_pred[i] == 1 else "Not Match"
        confidence = max(y_pred_proba[i]) * 100
        print(f"Text: {X_test[i][:50]}...")
        print(f"Actual: {actual}, Predicted: {pred} (Confidence: {confidence:.1f}%)")
        print("-" * 50)
    
    return accuracy

def main():
    goals = ["Amazon SDE", "ML Internship", "GATE ECE"]
    total_accuracy = 0
    
    print("Evaluating Resume Scorer Models")
    print("=" * 50)
    
    for goal in goals:
        accuracy = evaluate_goal(goal)
        total_accuracy += accuracy
    
    # Calculate average accuracy across all goals
    avg_accuracy = total_accuracy / len(goals)
    print("\n" + "=" * 50)
    print(f"Average Accuracy Across All Goals: {avg_accuracy:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
