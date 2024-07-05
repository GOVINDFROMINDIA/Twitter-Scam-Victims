import json
import networkx as nx
import pickle
import nltk
import re
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

def is_scam(input_text):
    input_text = preprocess_input(input_text)
    input_text_tfidf = vectorizer.transform([input_text])
    prediction = clf.predict(input_text_tfidf)
    predicted_label = labels[prediction[0]]
    return predicted_label

def is_scam2(text):
    text = preprocess_input(text)
    return is_scam(text)  # Return the result from the is_scam() function

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
    yes_count = sum(1 for tweet in tweets if is_scam2(tweet) == 'yes')
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

    # Add directed edges to the graph in reverse direction (from followers to following)
    for follower in connections:
        G.add_edge(str(follower), node)

# Remove self-loops from the graph
self_loops = list(nx.nodes_with_selfloops(G))
G.remove_edges_from(self_loops)

# Filter nodes with scam percentage above 60%
high_scam_nodes = [node for node, data in G.nodes(data=True) if data['scam_percentage'] > 60]

# Find hidden scammers
hidden_scammers = set()

for node in G.nodes:
    followers = list(G.predecessors(node))
    scam_followers = [follower for follower in followers if follower in high_scam_nodes]
    if len(scam_followers) >= 2:
        hidden_scammers.add(node)

# Combine high scam nodes and hidden scammers
all_relevant_nodes = set(high_scam_nodes) | hidden_scammers

# Create a subgraph with only relevant nodes
H = G.subgraph(all_relevant_nodes)

# Extract the scam percentages for nodes in the subgraph
scam_percentages = [H.nodes[node].get('scam_percentage', 0) for node in H.nodes]

# Use a colormap that ranges from light blue (low scam percentage) to dark blue (high scam percentage)
cmap = plt.get_cmap('Blues')

# Choose a layout algorithm
pos = nx.spring_layout(H, seed=42, k=0.15, iterations=50)

# Increase node size
node_size = 800

# Make edges less prominent
edge_color = 'gray'

# Plot the graph
plt.figure(figsize=(14, 10))

# Set node colors, red for hidden scammers and a color from the colormap for others
node_colors = []
for node in H.nodes:
    if node in hidden_scammers:
        node_colors.append('red')
    else:
        scam_percentage = H.nodes[node].get('scam_percentage', 0)
        node_colors.append(cmap(scam_percentage / 100))

# Draw nodes, edges, and labels
nx.draw(H, pos, with_labels=True, node_size=node_size, node_color=node_colors, font_size=12, arrowsize=15, edge_color=edge_color)
nx.draw_networkx_labels(H, pos, font_size=10, alpha=0.7)

# Adjust layout to prevent label overlap
plt.tight_layout()

# Display the plot
plt.title('NetworkX Directed Graph with Scammers in Blue and Hidden Scammers in Red')
plt.show()

# Print hidden scammer nodes
print("Hidden scammer nodes (followed by at least 2 high scam percentage nodes):")
for hidden_scammer in hidden_scammers:
    print(hidden_scammer)
