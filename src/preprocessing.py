import nltk
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data securely if missing
def download_nltk_data():
    import ssl
    
    # Bypass SSL verification for NLTK downloads if needed in some environments
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk_packages = ['wordnet', 'punkt', 'punkt_tab', 'omw-1.4']
    for package in nltk_packages:
        try:
            if package == 'wordnet':
                nltk.data.find('corpora/wordnet')
            else:
                nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_data()

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans and preprocesses the input text.
    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize
    4. Lemmatize
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Download stopwords if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    from nltk.corpus import stopwords
    from spellchecker import SpellChecker
    
    stop_words = set(stopwords.words('english'))
    spell = SpellChecker()
    
    # 4. Spell check, Remove stopwords, and Lemmatize
    processed_tokens = []
    for word in tokens:
        # Correct spelling if word is not known
        corrected_word = spell.correction(word)
        # spell.correction can return None if it cannot find a correction
        final_word = corrected_word if corrected_word else word
        
        if final_word not in stop_words or final_word in ['what', 'who', 'how']:
            lemmatized = lemmatizer.lemmatize(final_word)
            processed_tokens.append(lemmatized)
    
    # Rejoin tokens into a string for TF-IDF vectorizer
    return " ".join(processed_tokens)

if __name__ == "__main__":
    sample_text = "Hello!! Are you a Human? I am running fast."
    print(f"Original: {sample_text}")
    print(f"Cleaned:  {clean_text(sample_text)}")
