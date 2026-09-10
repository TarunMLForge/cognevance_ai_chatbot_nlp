# PLAN: AI Chatbot using NLP

## 1. Project Objective
Build a functional, professional, and testable AI chatbot utilizing classical Natural Language Processing (NLP) techniques and Machine Learning algorithms to classify user intent and generate appropriate responses, fulfilling the internship project requirements for Cognevance Technologies.

## 2. Official Requirement Mapping
| Official Requirement | Mapped Task | Phase |
| :--- | :--- | :--- |
| 1. Collect/create conversational data | Create a JSON intent dataset with varied examples | Phase 2 |
| 2. Preprocess text data | Implement lowercasing, tokenization, lemmatization | Phase 4 |
| 3. Use NLP/ML libraries | Utilize `nltk` and `scikit-learn` | Phase 4, 5, 6 |
| 4. Train chatbot to answer queries | Build training pipeline and train classifier | Phase 7 |
| 5. Implement intent recognition | Use a classification model (e.g., Logistic Regression) | Phase 6, 11 |
| 6. Implement response generation | Select responses based on predicted intent + confidence | Phase 9, 10 |
| 7. Build UI using Flask/Streamlit | Build `app/streamlit_app.py` for user interaction | Phase 12 |
| 8. Test and improve | Calculate accuracy, F1-score, and add unknown intent handling | Phase 8, 14 |

## 3. Functional Requirements
- Chatbot must accept user text input.
- Chatbot must classify the intent of the text.
- Chatbot must reply with a predefined response corresponding to the intent.
- Chatbot must handle unrecognized inputs gracefully (fallback/unknown intent).

## 4. Non-functional Requirements
- Reproducibility: Code must run reliably from a virtual environment.
- Modularity: ML logic must be separated from UI logic.
- Performance: Inference should be near-instantaneous (offline/local execution).
- Open Source: Entire stack must use free tools and libraries.

## 5. Dataset Strategy
We will create a structured, domain-specific **JSON intent dataset** containing tags (intents), patterns (user queries), and responses. It will feature intents for greetings, project info, help, capabilities, and a `movie_recommendation` intent to synergize with Project 1.

## 6. Free-Resource Strategy
- **Language/Environment:** Python 3.14 / venv
- **NLP Library:** `nltk` (Free, offline)
- **ML Library:** `scikit-learn` (Free, offline)
- **Interface:** `streamlit` (Free open-source web framework)
- **Deployment/Version Control:** Git & GitHub

## 7. NLP Approach
- **Cleaning:** Lowercasing, removing special characters/punctuation.
- **Tokenization:** Splitting text into words using `nltk`.
- **Lemmatization:** Reducing words to their base form using `nltk.WordNetLemmatizer`.
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) using `sklearn.feature_extraction.text.TfidfVectorizer` to capture word importance.

## 8. Model-Selection Reasoning
We will baseline multiple models (Logistic Regression, Linear SVM, Naive Bayes). 
- Deep learning is overkill for simple intent classification and requires more data to generalize. 
- Linear SVM or Logistic Regression typically perform exceptionally well on sparse TF-IDF vectors. We will evaluate and select the best one.

## 9. System Architecture
`User Message` -> `Text Cleaning` -> `Tokenization/Lemmatization` -> `TF-IDF Vectorization` -> `Classification Model` -> `Confidence Check` -> `Response Generation`.

## 10. Folder Structure
- `/data/`: Raw JSON intent files and validated outputs.
- `/src/`: Reusable Python modules (`preprocessing.py`, `model.py`, `chatbot.py`).
- `/app/`: Streamlit interface.
- `/models/`: Saved `.pkl` files (classifier, vectorizer, label encoder).
- `/tests/`: Automated unit and integration tests.
- `/outputs/`: Evaluation metrics and figures.

## 11. File-by-File Implementation Plan
- `data/raw/intents.json`: Training data.
- `src/preprocessing.py`: NLP functions.
- `src/features.py`: TF-IDF logic.
- `src/model.py`: Training and evaluation logic.
- `src/chatbot.py`: Inference engine (predict & respond).
- `app/streamlit_app.py`: UI.
- `tests/test_chatbot.py`: Test suite.

## 12. Development Phases
Follows the requested Phase 0 to Phase 25 strategy (Inspection -> Architecture -> Dataset -> Preprocessing -> Model -> Evaluation -> UI -> Tests -> Documentation).

## 13. Testing Strategy
- Unit tests for preprocessing to ensure consistent input/output.
- Integration test for end-to-end `chatbot.get_response(msg)`.
- Use `pytest` for test execution.

## 14. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (via `sklearn.metrics.classification_report`).
- Confusion matrix visualization to identify intent overlap.

## 15. UI Plan
- Streamlit application displaying a chat window interface.
- Includes a text input box and a scrollable chat history.
- Loads pre-trained model on startup using `@st.cache_resource`.

## 16. Documentation Plan
- Detailed `README.md` containing run instructions and architectural summary.
- Comprehensive `outputs/reports/project_report.md` for internship grading.

## 17. GitHub Submission Plan
- `.gitignore` configured properly.
- All code formatted and clean.
- Push to `cognevance_ai_chatbot_nlp`.

## 18. Risks and Mitigations
- **Risk:** Poor intent classification on unseen data.
  - **Mitigation:** Use robust lemmatization and TF-IDF, ensure diverse training patterns.
- **Risk:** Missing dependencies.
  - **Mitigation:** Strict `requirements.txt` generation and testing in a fresh virtual environment.

## 19. Definition of Done
Chatbot can reliably classify intent, Streamlit UI works locally, tests pass, no API keys are used, and all requested deliverables (Code, Report, README, Dataset) exist in the repository.
