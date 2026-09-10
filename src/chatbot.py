import random
import joblib
import json
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import clean_text

class ChatBot:
    def __init__(self, models_dir="models", dataset_path="data/raw/intents.json"):
        """Initializes the chatbot engine, loading all ML artifacts and response templates."""
        # Ensure paths exist
        if not os.path.exists(models_dir) or not os.path.exists(dataset_path):
            raise FileNotFoundError("Models or dataset missing. Ensure training has completed.")

        # Load the raw dataset to access the response arrays
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.intents_data = json.load(f)
            
        # Load serialized ML artifacts generated during Phase 7
        self.model = joblib.load(os.path.join(models_dir, 'intent_classifier.pkl'))
        self.vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        
        # Confidence threshold for fallback handling (Phase 10)
        # Using 0.40 because Linear SVM probability estimates can sometimes be distributed softly,
        self.CONFIDENCE_THRESHOLD = 0.35

    def get_response(self, user_input):
        """
        Process user input, predict intent, verify confidence, and generate response.
        (Phase 9, 10, 11)
        """
        # Validate empty input
        if not user_input or not user_input.strip():
            return "Please type a message so I can help you."

        # Preprocess exactly as in training
        cleaned_text = clean_text(user_input)
        
        # If input was only punctuation and became empty after cleaning
        if not cleaned_text:
            return "I'm sorry, I didn't catch any words in that. Could you rephrase?"

        # Vectorize
        vectorized_text = self.vectorizer.transform([cleaned_text])
        
        # Predict probability to get confidence score
        probabilities = self.model.predict_proba(vectorized_text)[0]
        max_prob_index = probabilities.argmax()
        confidence = probabilities[max_prob_index]
        
        # Get actual label
        predicted_label_index = self.model.predict(vectorized_text)[0]
        predicted_intent = self.label_encoder.inverse_transform([predicted_label_index])[0]
        
        # Check Confidence Threshold (Phase 10: Fallback Handling)
        if confidence < self.CONFIDENCE_THRESHOLD:
            return "I'm not fully sure I understood that. Could you rephrase your question?"
            
        # Select a random response from the matching intent (Phase 9: Response Generation)
        for intent in self.intents_data['intents']:
            if intent['intent'] == predicted_intent:
                responses = intent['responses']
                return random.choice(responses)
                
        # Failsafe
        return "I understood your intent, but I don't have a configured response for it."

if __name__ == "__main__":
    is_src = os.path.basename(os.getcwd()) == "src"
    models_dir = "../models" if is_src else "models"
    data_path = "../data/raw/intents.json" if is_src else "data/raw/intents.json"
    
    bot = ChatBot(models_dir=models_dir, dataset_path=data_path)
    
    print("Chatbot Initialized! Type 'quit' to exit.")
    while True:
        try:
            user_msg = input("\nYou: ")
            if user_msg.lower() in ['quit', 'exit']:
                break
            
            response = bot.get_response(user_msg)
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            break
