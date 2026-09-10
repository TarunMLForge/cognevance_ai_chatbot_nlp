import json
import pandas as pd
import os

def load_intents(filepath="data/raw/intents.json"):
    """Loads the raw JSON intents file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def validate_and_flatten_data(data):
    """
    Validates the dataset for empty patterns, duplicates, and missing intents.
    Returns a flattened pandas DataFrame suitable for training.
    """
    records = []
    seen_patterns = set()
    
    for intent_data in data.get("intents", []):
        tag = intent_data.get("intent", "").strip()
        patterns = intent_data.get("patterns", [])
        responses = intent_data.get("responses", [])
        
        if not tag:
            print("Warning: Found an intent block missing the 'intent' tag. Skipping.")
            continue
            
        if not patterns or not responses:
            print(f"Warning: Intent '{tag}' is missing patterns or responses.")
            
        for pattern in patterns:
            pattern = pattern.strip()
            if not pattern:
                print(f"Warning: Empty pattern found in intent '{tag}'.")
                continue
                
            pattern_lower = pattern.lower()
            if pattern_lower in seen_patterns:
                print(f"Warning: Duplicate pattern '{pattern}' found in intent '{tag}'.")
                continue
                
            seen_patterns.add(pattern_lower)
            records.append({
                "intent": tag,
                "text": pattern
            })
            
    df = pd.DataFrame(records)
    
    print("\n--- Data Quality Summary ---")
    print(f"Total Intents: {len(data.get('intents', []))}")
    print(f"Total Unique Patterns: {len(df)}")
    print("Class Distribution:")
    print(df['intent'].value_counts())
    print("----------------------------\n")
    
    return df

def save_processed_data(df, output_path="data/processed/flattened_intents.csv"):
    """Saves the flattened dataset to the processed folder."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

if __name__ == "__main__":
    is_src = os.path.basename(os.getcwd()) == "src"
    raw_path = "../data/raw/intents.json" if is_src else "data/raw/intents.json"
    processed_path = "../data/processed/flattened_intents.csv" if is_src else "data/processed/flattened_intents.csv"
    
    print("Loading raw dataset...")
    data = load_intents(raw_path)
    
    print("Validating dataset...")
    df = validate_and_flatten_data(data)
    
    print("Saving processed dataset...")
    save_processed_data(df, processed_path)
