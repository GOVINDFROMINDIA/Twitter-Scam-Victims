import re
import joblib
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (you only need to do this once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer for link prediction
link_model = joblib.load('voting_classifier.joblib')

# Load the trained model and vectorizer for scam text prediction
with open('check_spam_classifier.pkl', 'rb') as clf_file:
    scam_classifier = pickle.load(clf_file)

with open('check_spam_vectorizer.pkl', 'rb') as vectorizer_file:
    scam_vectorizer = pickle.load(vectorizer_file)

# Load labels from the text file for scam text prediction
with open('labels.txt', 'r') as labels_file:
    scam_labels = labels_file.read().splitlines()

# Define stopwords and lemmatizer for scam text prediction
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_input(text):
    # Preprocess the input text in the same way as the training data for scam text prediction
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def predict_link_type(link):
    link_prediction = link_model.predict([link])[0]
    
    # Check if the link type is defacement, phishing, or malware
    if link_prediction in ['defacement', 'phishing', 'malware']:
        return 1
    else:
        return 0

def predict_scam_text(input_text):
    # Preprocess the input text
    input_text = preprocess_input(input_text)
    
    # Vectorize the preprocessed text
    input_text_tfidf = scam_vectorizer.transform([input_text])
    
    # Make a prediction
    prediction = scam_classifier.predict(input_text_tfidf)
    
    # Convert prediction 'yes' to 1 and 'no' to 0
    if prediction[0] == 'yes':
        return 1
    else:
        return 0


def main(text):
    # Preprocess the input text
    text = preprocess_input(text)
    
    # Split text into links and texts
    # Here, I'm assuming links are URLs and texts are any other type of text
    links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    text_without_links = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Predict link types and scam text labels
    link_predictions = [predict_link_type(link) for link in links]
    scam_text_prediction = predict_scam_text(text_without_links)
    
    # Calculate average predictions
    avg_link_prediction = sum(link_predictions) / len(link_predictions) if link_predictions else 0
    avg_scam_text_prediction = scam_text_prediction
    
    # Calculate overall average
    overall_avg_prediction = (avg_link_prediction + avg_scam_text_prediction) / 2
    
    return overall_avg_prediction

# Example usage
text = "Get Free bitcoin http://www.garage-pirenne.be/index.php?option="
avg_prediction = main(text)
print("Average prediction:", avg_prediction)
