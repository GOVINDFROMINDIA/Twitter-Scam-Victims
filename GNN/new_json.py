import json
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Assuming your functions like preprocess_input and is_scam are already defined

# Load your classifier, vectorizer, and labels
with open('check_spam_classifier.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)

with open('check_spam_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('labels.txt', 'r') as labels_file:
    labels = labels_file.read().splitlines()

# Function to preprocess input text
def preprocess_input(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Function to predict if a text is a scam
def is_scam(input_text):
    input_text = preprocess_input(input_text)
    input_text_tfidf = vectorizer.transform([input_text])
    prediction = clf.predict(input_text_tfidf)
    predicted_label = labels[prediction[0]]
    return predicted_label

# Wrapper function (can be the same as is_scam)
def is_scam2(text):
    return is_scam(text)

# Load network data
with open('UniNetwork150.json', 'r') as json_file:
    data = json.load(json_file)

# Function to calculate scam labels for each node
def calculate_scam_labels(data):
    for node, info in data.items():
        tweets = info['Tweets']
        
        # Count scam tweets
        yes_count = sum(1 for tweet in tweets if is_scam2(tweet) == 'yes')
        total_tweets = len(tweets)
        scam_percentage = (yes_count / total_tweets) * 100 if total_tweets > 0 else 0

        # Assign labels based on scam percentage
        if scam_percentage <= 25:
            label = 0  # Low scam percentage
        elif 25 < scam_percentage <= 75:
            label = 1  # Medium scam percentage
        else:
            label = 2  # High scam percentage

        # Add the label to the node information
        info['Label'] = label

    return data

# Calculate labels and modify the data structure
labeled_data = calculate_scam_labels(data)

# Save the modified data to a new JSON file
with open('Labeled_UniNetwork150.json', 'w') as json_output:
    json.dump(labeled_data, json_output, indent=4)

print("New JSON file with labels has been saved as 'Labeled_UniNetwork150.json'")
