import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Ensure we can import from src regardless of where we run this
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import clean_text

def train_and_evaluate(df):
    """
    Trains multiple intent classification models and selects the best one.
    Steps:
    1. Preprocess text using NLTK pipeline.
    2. Extract TF-IDF features (unigrams and bigrams).
    3. Encode intent labels.
    4. Train Logistic Regression, SVM, and Naive Bayes.
    5. Evaluate on training data (since dataset is small).
    6. Save the best model, vectorizer, and label encoder.
    """
    print("Preprocessing text data...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("Extracting TF-IDF features...")
    # TF-IDF prioritizes unique informative words over common stop words
    vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 
    X = vectorizer.fit_transform(df['clean_text'])
    
    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df['intent'])
    
    # We compare 3 classical ML models suitable for text classification
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Naive Bayes": MultinomialNB()
    }
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            # We select the model that performs best
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nSelected Best Model: {best_name} with Accuracy: {best_acc:.4f}")
    
    # Generate full report for the best model
    print("\n--- Final Classification Report ---")
    y_pred_best = best_model.predict(X)
    print(classification_report(y, y_pred_best, target_names=le.classes_))
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y, y_pred_best)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {best_name}')
    plt.xlabel('Predicted Intent')
    plt.ylabel('Actual Intent')
    plt.tight_layout()
    
    is_src = os.path.basename(os.getcwd()) == "src"
    fig_dir = "../outputs/figures" if is_src else "outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'confusion_matrix.png'))
    print(f"Saved confusion matrix to {fig_dir}/confusion_matrix.png")
    
    # Save artifacts for inference
    model_dir = "../models" if is_src else "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, 'intent_classifier.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))
    print(f"Saved model artifacts to {model_dir}/")

if __name__ == "__main__":
    is_src = os.path.basename(os.getcwd()) == "src"
    data_path = "../data/processed/flattened_intents.csv" if is_src else "data/processed/flattened_intents.csv"
    
    df = pd.read_csv(data_path)
    train_and_evaluate(df)
