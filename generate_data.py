import random, networkx as nx, json
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate shortest path dataset')
    parser.add_argument('--max_nodes', type=int, default=8,
                      help='Maximum number of nodes in generated graphs')
    parser.add_argument('--p', type=float, default=0.4,
                      help='Edge probability for random graph generation')
    parser.add_argument('--train_samples', type=int, default=47500,
                      help='Number of training examples to generate')
    parser.add_argument('--test_samples', type=int, default=2500,
                      help='Number of test examples to generate')
    parser.add_argument('--w_range', type=int, nargs=2, default=[1, 9],
                      help='Range for edge weights (min max)')
    return parser.parse_args()

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
    # Get all shortest paths and choose the lexicographically smallest one
    all_paths = list(nx.all_shortest_paths(g, s, t, weight='weight'))
    sp_nodes = min(all_paths, key=lambda p: [str(x) for x in p])
    sp_len   = nx.shortest_path_length(g, s, t, weight='weight')
    return {
        "input": f"edges: {edge_txt}; start:{s}; goal:{t}",
        "label": f"path: {'→'.join(str(node) for node in sp_nodes)}, length: {sp_len}"
    }

def main():
    args = parse_args()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate train data
    train_out = data_dir / "shortest_paths_train.jsonl"
    print(f"Generating {args.train_samples} training examples...")
    with train_out.open("w") as f:
        for _ in range(args.train_samples):
            json.dump(encode_example(random_graph(args.max_nodes, args.p, args.w_range)), f)
            f.write("\n")

    # Generate test data
    test_out = data_dir / "shortest_paths_test.jsonl"
    print(f"Generating {args.test_samples} test examples...")
    with test_out.open("w") as f:
        for _ in range(args.test_samples):
            json.dump(encode_example(random_graph(args.max_nodes, args.p, args.w_range)), f)
            f.write("\n")
    
    print(f"Data generation complete. Files saved in {data_dir}/")

if __name__ == "__main__":
    main()