import random, networkx as nx, json
from pathlib import Path

def random_graph(max_nodes=8, p=0.4, w_range=(1,9)):
    n = random.randint(4, max_nodes)
    g = nx.gnp_random_graph(n, p, directed=False)
    # ensure connectivity
    while not nx.is_connected(g):
        g = nx.gnp_random_graph(n, p, directed=False)
    for (u,v) in g.edges:
        g.edges[u,v]['weight'] = random.randint(*w_range)
    return g

def encode_example(g):
    nodes = list(g.nodes)
    s, t = random.sample(nodes, 2)
    edge_txt = ",".join(f"{u}-{v}:{g.edges[u,v]['weight']}"
                        for u,v in g.edges)
    sp_nodes = nx.shortest_path(g, s, t, weight='weight')
    sp_len   = nx.shortest_path_length(g, s, t, weight='weight')
    return {
        "input": f"edges: {edge_txt}; start:{s}; goal:{t}",
        "label": f"path: {'→'.join(str(node) for node in sp_nodes)}, length: {sp_len}"
    }

# Generate train data
train_out = Path("shortest_paths_train.jsonl").open("w")
for _ in range(47_500):  # 95% of data for training
    json.dump(encode_example(random_graph()), train_out)
    train_out.write("\n")
train_out.close()

# Generate test data
test_out = Path("shortest_paths_test.jsonl").open("w")
for _ in range(2_500):  # 5% of data for testing
    json.dump(encode_example(random_graph()), test_out)
    test_out.write("\n")
test_out.close()