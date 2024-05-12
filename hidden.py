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

# Initialize a dictionary to store followers of each node
followers = {}

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

    # Store followers for each node
    followers[node] = connections

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

# Extract normal nodes (less than 70% scam rate)
normal_nodes = [node for node in G.nodes if G.nodes[node]['scam_percentage'] < 70]

# Initialize a list to store nodes that follow at least one scammer
valid_normal_nodes = []

# Iterate through normal nodes
for node in normal_nodes:
    # Check if the node exists in the graph
    if node not in G.nodes:
        continue

    # Check how many scammers are followed by the current normal node
    for follower in followers[node]:
        # Check if the follower exists in the graph
        if follower in G.nodes and G.nodes[follower]['scam_percentage'] > 70:
            # Add the node to the list of valid normal nodes and break the loop
            valid_normal_nodes.append(node)
            break

# Create a subgraph containing only valid normal nodes and their edges
valid_normal_subgraph = G.subgraph(valid_normal_nodes + [node for node in G.nodes if node in valid_normal_nodes])

# Check if the subgraph contains nodes and edges before plotting
if valid_normal_subgraph.number_of_nodes() == 0 or valid_normal_subgraph.number_of_edges() == 0:
    print("No nodes or edges to plot.")
else:
    # Choose a layout algorithm
    pos = nx.spring_layout(valid_normal_subgraph, seed=42, k=0.1, iterations=50)

    # Increase node size
    node_size = 800

    # Make edges less prominent
    edge_color = 'gray'

    # Make node labels semi-transparent
    nx.draw_networkx_labels(valid_normal_subgraph, pos, font_size=10, alpha=0.7)

    # Close all existing figures
    plt.close('all')

    # Create a new figure
    plt.figure(figsize=(14, 10))

    # Plot the subgraph
    nx.draw(valid_normal_subgraph, pos, with_labels=True, node_size=node_size, font_size=12, arrowsize=15, edge_color=edge_color)

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Display the plot
    plt.title('NetworkX Directed Graph with Valid Normal Nodes Following Scammers')
    plt.show()
