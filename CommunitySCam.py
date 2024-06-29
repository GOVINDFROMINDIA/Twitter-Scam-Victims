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


def is_scam(text):
    text = preprocess_input(text)
    input_text_tfidf = vectorizer.transform([text])
    prediction = clf.predict(input_text_tfidf)
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

        # Add directed edges to the graph in reverse direction (from followers to following)
        for follower in connections:
            G.add_edge(str(follower), node)

# Remove self-loops from the graph
self_loops = list(nx.nodes_with_selfloops(G))
G.remove_edges_from(self_loops)

# Choose a layout algorithm
pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

# Increase node size
node_size = 800

# Make edges less prominent
edge_color = 'gray'

# Make node labels semi-transparent
nx.draw_networkx_labels(G, pos, font_size=10, alpha=0.7)

# Close all existing figures
plt.close('all')

# Create a new figure
plt.figure(figsize=(14, 10))

# Plot the graph
nx.draw(G, pos, with_labels=True, node_size=node_size, font_size=12, arrowsize=15, edge_color=edge_color)

# Adjust layout to prevent label overlap
plt.tight_layout()

# Display the plot
plt.title('NetworkX Directed Graph of Hidden Scam Communities (Threshold: 70%)')
plt.show()

# Print top nodes with the highest scam percentage
sorted_nodes = sorted(G.nodes, key=lambda x: yes_counts[x], reverse=True)
print("\nTop nodes with the highest scam percentage:")
for node in sorted_nodes[:10]:
    print(f"Node {node}: 'Yes' Count = {yes_counts[node]}, 'No' Count = {no_counts[node]}")

# Print bottom nodes with the lowest scam percentage
print("\nBottom nodes with the lowest scam percentage:")
for node in sorted_nodes[-10:]:
    print(f"Node {node}: 'Yes' Count = {yes_counts[node]}, 'No' Count = {no_counts[node]}")
