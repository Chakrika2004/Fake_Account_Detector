import pandas as pd
import networkx as nx
from community import community_louvain

def load_data():
    df = pd.read_csv("../data/fake_accounts_dataset.csv")
    edges = pd.read_csv("../data/network_connections.csv")
    return df, edges

def create_graph(edges):
    graph = nx.from_pandas_edgelist(edges, 'source', 'target', create_using=nx.Graph())
    communities = community_louvain.best_partition(graph)
    betweenness = nx.betweenness_centrality(graph)
    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    return graph, communities, betweenness, eigenvector
