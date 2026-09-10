# AI Chatbot using NLP 🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## 📌 Project Overview
This repository contains an AI-powered conversational chatbot built for the Cognevance Technologies Machine Learning Internship (Project 2). It uses classical Natural Language Processing (NLP) to classify user intents and generate appropriate responses, focusing on a lightweight, offline-first approach without relying on paid APIs or expensive LLMs.

## 🎯 Problem Statement
Organizations often require automated assistants to handle repetitive inquiries. This project demonstrates how to build a fully functional, open-source intent classification system that parses text, extracts features, predicts the user's goal, and responds intelligently with a failsafe mechanism for unknown queries.

## 📊 Dataset
The model is trained on a custom JSON dataset (`data/raw/intents.json`) consisting of various conversational intents (Greetings, Help, Internship Info, Contact, Capabilities, Movie Recommendations).

## 🛠 Free Resources Used
- **Language:** Python
- **NLP:** NLTK (Tokenization, Lemmatization)
- **ML:** Scikit-Learn (TF-IDF, Logistic Regression)
- **Interface:** Streamlit (Free open-source web framework)
- **Evaluation:** Matplotlib / Seaborn

## 🚀 System Architecture
1. **NLP Preprocessing:** Lowercasing, punctuation stripping, tokenization, and lemmatization via NLTK WordNet.
2. **Feature Extraction:** TF-IDF Vectorization capturing n-grams (1, 2) to weigh words by their informativeness.
3. **Intent Classification:** A Logistic Regression model trained to predict intent labels.
4. **Response Generation:** Randomly selecting a predefined response for the predicted intent.
5. **Confidence/Fallback Handling:** Queries scoring below a 25% confidence threshold trigger a safe fallback response.

## 📂 Project Structure
```text
cognevance_ai_chatbot_nlp/
│
├── data/
│   ├── raw/intents.json             # Training intent dataset
│   └── processed/                   # Validated CSV flattened data
├── models/                          # Serialized ML artifacts (.pkl)
├── src/                   
│   ├── data_loader.py               # Dataset validation script
│   ├── preprocessing.py             # NLP cleaning pipeline
│   ├── model.py                     # ML training pipeline
│   └── chatbot.py                   # Inference engine
├── app/
│   └── streamlit_app.py             # Web interface
├── outputs/
│   ├── figures/                     # Confusion matrix visualization
│   └── reports/                     # Project report
├── tests/
│   └── test_chatbot.py              # Automated unit/integration tests
├── requirements.txt                 # Project dependencies
└── README.md                        # You are here
```

## ⚙️ Installation & Virtual Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cognevance_ai_chatbot_nlp.git
   cd cognevance_ai_chatbot_nlp
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the Chatbot

**1. Prepare Data & Train Model:**
```bash
python src/data_loader.py
python src/model.py
```
*This validates the JSON, extracts TF-IDF vectors, trains the classifier, and saves the artifacts.*

**2. Launch the Streamlit Interface:**
```bash
streamlit run app/streamlit_app.py
```

## 🧪 Running Tests
We use Python's built-in `unittest` framework to ensure high code quality and prevent regression:
```bash
python -m unittest tests/test_chatbot.py
```

## 📈 Results
The Logistic Regression intent classifier achieved an accuracy of **95.5%** on the validation set. Detailed evaluation metrics and confusion matrices can be found in `outputs/reports/project_report.md`.

## 🔮 Future Improvements
- **Contextual Memory:** Implementing a state-tracker to remember past turns in the conversation.
- **Entity Extraction:** Using SpaCy NER to extract names or dates from the user's message.
- **Deep Learning:** Exploring intent classification with lightweight neural networks (e.g., FastText or an embedded LSTM) for larger datasets.

## 👨‍💻 Author
Developed for Cognevance Technologies Artificial Intelligence & Machine Learning Internship.
