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
    yes_count = sum(1 for tweet in tweets if is_scam(tweet) == 'yes')
    total_tweets = len(tweets)
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

# Use a custom color palette for nodes
node_colors = [cmap(scam_percentage / 100) for scam_percentage in scam_percentages]

# Plot the graph
nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_colors, font_size=12, arrowsize=15, edge_color=edge_color)

# Adjust layout to prevent label overlap
plt.tight_layout()

# Display the plot
plt.title('NetworkX Directed Graph with Node Color by Scam Percentage')
plt.show()

# Print top 10 nodes with the highest scam percentage
sorted_nodes = sorted(G.nodes, key=lambda x: G.nodes[x].get('scam_percentage', 0), reverse=True)
print("\nTop 10 nodes with the highest scam percentage:")
for node in sorted_nodes[:10]:
    scam_percentage = G.nodes[node].get('scam_percentage', 0)
    print(f"Node {node}: Scam Percentage = {scam_percentage:.2f}%, 'Yes' Count = {yes_counts[node]}, 'No' Count = {no_counts[node]}")

# Print bottom 10 nodes with the lowest scam percentage
print("\nBottom 10 nodes with the lowest scam percentage:")
for node in sorted_nodes[-10:]:
    scam_percentage = G.nodes[node].get('scam_percentage', 0)
    print(f"Node {node}: Scam Percentage = {scam_percentage:.2f}%, 'Yes' Count = {yes_counts[node]}, 'No' Count = {no_counts[node]}")
