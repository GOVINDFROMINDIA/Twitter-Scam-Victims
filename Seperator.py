import re
import joblib
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (you only need to do this once)
nltk.download('stopwords')
nltk.download('wordnet')
#import nltk
nltk.download('omw-1.4')

# Load the trained model and vectorizer for link prediction
link_model = joblib.load('voting_classifier.joblib')

with open('check_spam_classifier.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)

with open('check_spam_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load labels from the text file
with open('labels.txt', 'r') as labels_file:
    labels = labels_file.read().splitlines()

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
    input_text_tfidf = vectorizer.transform([input_text])
    
    # Make a prediction
    prediction = clf.predict(input_text_tfidf)
    
    # Get the label using the labels list
    predicted_label = labels[prediction[0]]
    # Convert prediction 'yes' to 1 and 'no' to 0
    if predicted_label == 'yes':
        return 1
    else:
        return 0    


# OR Gate Function
def OR_Gate(x1, y1):
    if x1 == 1 or y1 == 1:
        return 1
    else:
        return 0

def is_scam(text):
    # Preprocess the input text
    text = preprocess_input(text)
    
    # Split text into links and texts
    # Here, I'm assuming links are URLs and texts are any other type of text
    links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    text_without_links = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    print(links)
    print(text_without_links)
    # Predict link types and scam text labels
    link_predictions = [predict_link_type(link) for link in links]
    scam_text_prediction = predict_scam_text(text_without_links)
    print(scam_text_prediction)
    
    # Pass predictions to OR_Gate function
    result = OR_Gate(scam_text_prediction, max(link_predictions, default=0))
    return result

# Example usage
text = "megha is for sale http://www.amazin.com" 
output = is_scam(text)

print("Output:", output)
