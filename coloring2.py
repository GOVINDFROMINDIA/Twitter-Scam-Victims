import json
import networkx as nx
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

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


def is_scam(input_text):
    input_text = preprocess_input(input_text)
    input_text_tfidf = vectorizer.transform([input_text])
    prediction = clf.predict(input_text_tfidf)
    predicted_label = labels[prediction[0]]
    return predicted_label

with open('network.json', 'r') as json_file:
    data = json.load(json_file)

G = nx.DiGraph()  # Use DiGraph instead of Graph
scam_percentages = {}
yes_counts = {}
no_counts = {}

# Iterate through the data and add nodes and directed edges
for node, info in data.items():
    connections = info['Connections']
    tweets = info['Tweets']

    # Initialize counts for scam tweets, "yes" and "no"
    scam_count = 0
    yes_count = 0
    no_count = 0
    total_tweets = len(tweets)

    for tweet in tweets:
        if is_scam(tweet) == 'yes':
            yes_count += 1

    # Count "yes" based on the difference between total tweets and "no" count
    no_count = total_tweets - yes_count

    # Calculate the scam percentage
    scam_percentage = (yes_count / total_tweets) * 100 if total_tweets > 0 else 0

    # Store counts of "yes" and "no" for each node
    yes_counts[node] = yes_count
    no_counts[node] = no_count

    # Add the node to the graph
    G.add_node(node)

    # Store the scam percentage as a node attribute
    G.nodes[node]['scam_percentage'] = scam_percentage

    # Add directed edges to the graph
    for neighbor in connections:
        G.add_edge(node, str(neighbor))

# Remove self-loops from the graph
self_loops = list(nx.nodes_with_selfloops(G))
G.remove_edges_from(self_loops)

# Extract the scam percentages for nodes
scam_percentages = [G.nodes[node].get('scam_percentage', 0) for node in G.nodes]

# Use a colormap that ranges from light blue (low scam percentage) to dark blue (high scam percentage)
cmap = plt.get_cmap('Blues')
node_colors = [cmap(scam_percentage / 100) for scam_percentage in scam_percentages]

# Plot the NetworkX directed graph with node colors based on scam percentage
pos = nx.spring_layout(G)  # Layout the nodes

fig, ax = plt.subplots()
nx.draw(G, pos, with_labels=True, node_size=300, node_color=node_colors, font_size=10, ax=ax, arrowsize=15)
plt.title('NetworkX Directed Graph Based on Connections (Self-loops removed) with Node Color by Scam Percentage')
plt.show()

print("Scam percentages, counts of 'yes', and counts of 'no' for each node:")
for node in G.nodes:
    scam_percentage = G.nodes[node].get('scam_percentage', 0)
    print(
        f"Node {node}: Scam Percentage = {scam_percentage:.2f}%, 'Yes' Count = {yes_counts[node]}, 'No' Count = {no_counts[node]}")
