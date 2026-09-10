# Project Report: AI Chatbot using NLP

## 1. Problem Statement
Automated assistants are increasingly necessary to scale customer support and handle repetitive inquiries. However, relying on expensive external generative AI APIs (like OpenAI) is not always feasible due to cost, latency, or data privacy concerns. This project demonstrates how to build a highly accurate, entirely offline intent-classification chatbot using classical Natural Language Processing (NLP).

## 2. Objective
The objective is to build an end-to-end NLP chatbot that parses user input, correctly classifies the underlying intent, and returns an appropriate predefined response. It must include fallback logic for unrecognized queries and provide a clean, professional web interface for user interaction.

## 3. Dataset
The model relies on a custom-designed JSON dataset (`data/raw/intents.json`). It maps 8 distinct intents (such as greetings, goodbyes, requests for help, and project capabilities) to a variety of user query patterns and system responses. This format allows for rapid scaling and easy maintenance.

## 4. NLP Preprocessing
Before feature extraction, raw text is passed through an NLTK-powered pipeline:
1. **Lowercasing:** Standardizes capitalization.
2. **Punctuation Removal:** Strips noise from the text.
3. **Tokenization:** Splits sentences into discrete words.
4. **Lemmatization:** Reduces words to their morphological root (e.g., "running" becomes "run") using the `WordNetLemmatizer`. This is crucial for small datasets as it groups variations of the same word together.

## 5. Feature Extraction
We utilized `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) capturing unigrams and bigrams. TF-IDF evaluates how relevant a word is to a specific intent by penalizing overly common words (like "the") and boosting words that strongly signal an intent (like "help" or "contact").

## 6. Intent Classification
We established a training pipeline to evaluate three classical Machine Learning algorithms: Logistic Regression, Linear SVM, and Naive Bayes.
- **Winner:** Logistic Regression achieved **95.56% Accuracy**.
- Classical ML was chosen over Deep Learning because neural networks easily overfit small, domain-specific intent datasets. Logistic Regression provides robust, interpretable decision boundaries on sparse TF-IDF matrices.

## 7. Response Generation & Fallback Logic
During inference, the `ChatBot` engine applies the exact same preprocessing and vectorization steps to the user's input.
- If the Logistic Regression model predicts an intent with a confidence score **below 25%**, the chatbot triggers a fallback response: *"I'm not fully sure I understood that. Could you rephrase your question?"*
- If the confidence is above the threshold, it randomly selects one of the predefined responses associated with that intent.

## 8. Chatbot Interface
We built the user interface using **Streamlit**. It provides an elegant, reactive chat window that mimics real messaging applications. The ML models are loaded into memory once using Streamlit caching (`@st.cache_resource`), resulting in near-instantaneous response times.

## 9. Testing & Quality Assurance
The codebase is covered by automated unit tests using Python's `unittest` framework. The tests verify:
- Accurate lemmatization.
- Empty and punctuation-only inputs triggering the correct fallbacks.
- Correct intent matching for standard greetings.
- Safe fallback behavior for out-of-domain queries.

## 10. Conclusion
This project successfully fulfills all requirements for the Cognevance ML Internship Project 2. By leveraging NLTK and Scikit-Learn, we built a highly capable, offline, and free-to-run conversational agent architecture that can easily be expanded with new intents and integrated into larger software systems.
