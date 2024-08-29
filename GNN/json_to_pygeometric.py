import json
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
import numpy as np

# Load your JSON dataset
with open('Labeled_UniNetwork150.json', 'r') as json_file:
    data = json.load(json_file)

# Convert tweets to TF-IDF features
tweets_list = [" ".join(info["Tweets"]) for node, info in data.items()]
vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
node_features = vectorizer.fit_transform(tweets_list).toarray()

# Create node labels
node_labels = torch.tensor([info["Label"] for node, info in data.items()], dtype=torch.long)

# Create edges (followers relationships)
edges = []
for node, info in data.items():
    for follower in info["Followers"]:
        edges.append([int(node) - 1, follower - 1])  # Adjusting to zero-indexing

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Convert to PyTorch Geometric Data format
data = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=edge_index, y=node_labels)

# Save or use the data
torch.save(data, 'processed_data.pt')  # Save the processed data
print(data)
