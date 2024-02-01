import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (you only need to do this once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
with open('check_spam_classifier.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)

with open('check_spam_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load labels from the text file
with open('labels.txt', 'r') as labels_file:
    labels = labels_file.read().splitlines()

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_input(text):
    # Preprocess the input text in the same way as the training data
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def is_scam(input_text):
    # Preprocess the input text
    input_text = preprocess_input(input_text)
    
    # Vectorize the preprocessed text
    input_text_tfidf = vectorizer.transform([input_text])
    
    # Make a prediction
    prediction = clf.predict(input_text_tfidf)
    
    # Get the label using the labels list
    predicted_label = labels[prediction[0]]
    
    return predicted_label

if __name__ == "__main__":
    user_input = input("Enter text to check if it's a scam: ")
    result = is_scam(user_input)
    print(f"Predicted label: {result}")
