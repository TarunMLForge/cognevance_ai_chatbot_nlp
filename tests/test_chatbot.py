import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import clean_text
from src.chatbot import ChatBot

class TestChatbotEngine(unittest.TestCase):
    
    def setUp(self):
        # We assume the model has been trained for these tests
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'intents.json'))
        self.bot = ChatBot(models_dir=models_dir, dataset_path=dataset_path)
        
    def test_preprocessing(self):
        """Test NLP cleaning, tokenization, and lemmatization."""
        raw = "Hello! Are you running?"
        # WordNet defaults to noun pos, so 'running' stays 'running'. 'are' and 'you' are removed as stopwords.
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "hello running")
        
    def test_empty_input(self):
        """Test empty/whitespace input fallback."""
        response = self.bot.get_response("   ")
        self.assertEqual(response, "Please type a message so I can help you.")
        
    def test_punctuation_only_input(self):
        """Test punctuation-only input fallback."""
        response = self.bot.get_response("???!!!")
        self.assertEqual(response, "I'm sorry, I didn't catch any words in that. Could you rephrase?")
        
    def test_valid_intent(self):
        """Test a clear intent classification (Greeting)."""
        response = self.bot.get_response("Hello there!")
        print(f"DEBUG Greeting Response: {response}")
        is_valid_greeting = any(r in response for r in ["Hello", "Hi", "Greetings"])
        self.assertTrue(is_valid_greeting, "Bot did not return a valid greeting response.")
        
    def test_typo_resilience(self):
        """Test that the chatbot handles common misspellings gracefully."""
        # Using a deliberate typo ('reccomend')
        response = self.bot.get_response("can you reccomend a movie")
        is_movie_response = any(r in response for r in ["Project 1", "recommend", "Matrix"])
        self.assertTrue(is_movie_response, "Bot failed to recognize movie recommendation despite typos.")
        
    def test_unknown_fallback(self):
        """Test the low-confidence fallback."""
        # A completely random string that shouldn't match any intent with high confidence
        response = self.bot.get_response("I want to order a pepperoni pizza with extra cheese")
        self.assertEqual(response, "I'm not fully sure I understood that. Could you rephrase your question?")

if __name__ == '__main__':
    unittest.main()
