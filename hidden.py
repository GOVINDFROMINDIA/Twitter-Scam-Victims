import json
import networkx as nx
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

# Load classifier and vectorizer
with open('check_spam_classifier.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)

with open('check_spam_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('labels.txt', 'r') as labels_file:
    labels = labels_file.read().splitlines()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_input(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def predict_scam_text(input_text):
    # Preprocess the input text
    input_text = preprocess_input(input_text)

    # Vectorize the preprocessed text
    input_text_tfidf = vectorizer.transform([input_text])

    # Make a prediction
    prediction = clf.predict(input_text_tfidf)

    # Get the label using the labels list
    predicted_label = labels[prediction[0]]

    return predicted_label

# Load network data
with open('UniNetwork150.json', 'r') as json_file:
    data = json.load(json_file)

# Create a directed graph
G = nx.DiGraph()

# Initialize dictionaries to store counts of 'yes' and 'no' for each node
yes_counts = {}
no_counts = {}

# Iterate through the data and add nodes and directed edges
for node, info in data.items():
    connections = info['Followers']
    tweets = info['Tweets']

    # Initialize counts for scam tweets, "yes" and "no"
    yes_count = sum(1 for tweet in tweets if predict_scam_text(tweet) == 'yes')
    total_tweets = len(tweets)
    no_count = total_tweets - yes_count

    # Calculate the scam percentage
    scam_percentage = (yes_count / total_tweets) * 100 if total_tweets > 0 else 0

    # Store counts of "yes" and "no" for each node
    yes_counts[node] = yes_count
    no_counts[node] = no_count

    # Add the node to the graph if it belongs to a hidden scam community
    if scam_percentage > 70:
        G.add_node(node)

        # Add directed edges to the graph
        for neighbor in connections:
            G.add_edge(node, str(neighbor))

# Remove self-loops from the graph
self_loops = list(nx.nodes_with_selfloops(G))
G.remove_edges_from(self_loops)

# Find hidden scammer nodes
hidden_scammer_nodes = set()
for node in G.nodes:
    for neighbor in G.successors(node):
        # Check if the followed node has a scam percentage less than 25%
        if neighbor in yes_counts:
            scam_percentage = (yes_counts[neighbor] / (yes_counts[neighbor] + no_counts[neighbor])) * 100 if (yes_counts[neighbor] + no_counts[neighbor]) > 0 else 0
            if scam_percentage < 25:
                hidden_scammer_nodes.add(neighbor)

# Print hidden scammer nodes
print("\nHidden scammer nodes with scam percentage less than 25% that are followed by scam nodes:")
for node in hidden_scammer_nodes:
    scam_percentage = (yes_counts[node] / (yes_counts[node] + no_counts[node])) * 100 if (yes_counts[node] + no_counts[node]) > 0 else 0
    print(f"Node {node}: 'Yes' Count = {yes_counts[node]}, 'No' Count = {no_counts[node]}, Scam Percentage = {scam_percentage:.2f}%")

# Plot the graph with hidden scammer nodes highlighted
pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)
plt.figure(figsize=(14, 10))
nx.draw(G, pos, with_labels=True, node_size=800, font_size=12, arrowsize=15, edge_color='gray')
nx.draw_networkx_nodes(G, pos, nodelist=hidden_scammer_nodes, node_color='red')
plt.title('NetworkX Directed Graph of Hidden Scam Communities with Hidden Scammer Nodes Highlighted')
plt.show()
