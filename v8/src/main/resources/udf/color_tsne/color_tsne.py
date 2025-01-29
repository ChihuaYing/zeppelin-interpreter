import os
import pickle
import numpy as np

from pymilvus import connections, Collection
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tqdm


class MilvusDao:
    def __init__(self, kvargs):
        milvus_host = "localhost"
        milvus_port = 19530
        if kvargs.get("host"):
            milvus_host = kvargs["host"].decode("utf-8")
        if kvargs.get("port"):
            milvus_port = int(kvargs["port"].decode("utf-8"))

        self.batch_size = 1000
        print("Connecting to Milvus")
        connections.connect(host=milvus_host, port=milvus_port)
        self.collection = Collection(name='embeddings')

    def fetch_all_embeddings(self):
        print("fetching all embeddings")
        iter = self.collection.query_iterator(
            output_fields=["path", "embedding"],
        )
        while True:
            entities = iter.next()
            if not entities:
                iter.close()
                break

            for entity in entities:
                yield entity["path"], entity["embedding"]


def build_path_tree(nodes_list: list[str]) -> dict:
    tree = {}
    for nodes in nodes_list:
        current_node = tree
        for node in nodes.split("."):
            current_node = current_node.setdefault(node, {})
    return tree

def allocate_colors(tree: dict, low:float=0.0, high:float=1.0, colors:dict[str,float]={},path:list[str]=[]) -> dict:
    children = list(tree.keys())
    n_children = len(children)
    children_low = [low + i * (high - low) / n_children for i in range(n_children)]
    children_high = [low + (i + 1) * (high - low) / n_children for i in range(n_children)]
    for child, child_low, child_high in zip(children, children_low, children_high):
        path.append(child)
        colors[".".join(path)] = (child_low + child_high) / 2
        allocate_colors(tree[child], child_low, child_high, colors, path)
        path.pop()

    return colors

class TSNEVisualizer:

    @staticmethod
    def cache_get(path, supplier):
        path = "cache/" + path + ".pkl"
        if not os.path.exists(path):
            print(f"Cache not found, fetching data {path}")
            data = supplier()
            with open(path, "wb") as f:
                pickle.dump(data, f)
        with open(path, "rb") as f:
            return pickle.load(f)

    def __init__(self):
        self.embeddings = {
            # path: embedding
            # for path, embedding in tqdm.tqdm(MilvusDao({}).fetch_all_embeddings(), desc="Fetching paths")
        }
        self.paths= list(sorted(self.embeddings.keys()))
        tree = build_path_tree(self.paths)
        self.colors = allocate_colors(tree)

    def fit_tsne(self,embeddings:np.array, perplexity:int =30):
        tsne = TSNE(random_state=0, max_iter=5000, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(embeddings)
        return tsne_embeddings.T

    def get_cmaps(self):
        return [
            "Dark2",
        ]

    def visualize_2d(self, tsne_embeddings, colors, s):
        for cmap in self.get_cmaps():
            plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
            plt.scatter(tsne_embeddings[0], tsne_embeddings[1], c=colors, cmap=cmap, s=s)
            plt.tight_layout()

if __name__ == '__main__':
    visualizer = TSNEVisualizer()
    print("Paths:", len(visualizer.embeddings))
    print("Fitting TSNE with 2 components for 1-level paths")
    l1_paths = [path for path in visualizer.paths if len(path.split(".")) == 1]
    l1_embeddings = np.array([visualizer.embeddings[path] for path in l1_paths])
    l1_colors = TSNEVisualizer.cache_get("l1_colors", lambda: [visualizer.colors[path] for path in l1_paths])
    for l1_perplexity in [50]:
        l1d2 = TSNEVisualizer.cache_get(f"color_l1d2_{l1_perplexity}", lambda: visualizer.fit_tsne(l1_embeddings, perplexity=l1_perplexity))
        visualizer.visualize_2d(l1d2, l1_colors,4)
    # plt.show()
    plt.savefig('sources.png', dpi=1200)
    plt.clf()
    print("Fitting TSNE with 2 components for 2-level paths")
    l2_paths = [path for path in visualizer.paths if len(path.split(".")) == 2]
    l2_embeddings = np.array([visualizer.embeddings[path] for path in l2_paths])
    l2_colors = TSNEVisualizer.cache_get("l2_colors", lambda: [visualizer.colors[path] for path in l2_paths])
    for l2_perplexity in [55]:
        l2d2 = TSNEVisualizer.cache_get(f"color_l2d2_{l2_perplexity}", lambda: visualizer.fit_tsne(l2_embeddings, perplexity=l2_perplexity))
        visualizer.visualize_2d(l2d2, l2_colors,0.4)
    # plt.show()
    plt.savefig('tables.png', dpi=1200)
    plt.clf()
    print("Fitting TSNE with 2 components for 3-level paths")
    l3_paths = [path for path in visualizer.paths if len(path.split(".")) == 3]
    l3_embeddings = np.array([visualizer.embeddings[path] for path in l3_paths])
    l3_colors = TSNEVisualizer.cache_get("l3_colors", lambda: [visualizer.colors[path] for path in l3_paths])
    for l3_perplexity in [55]:
        l3d2 = TSNEVisualizer.cache_get(f"color_l3d2_{l3_perplexity}", lambda: visualizer.fit_tsne(l3_embeddings, perplexity=l3_perplexity))
        visualizer.visualize_2d(l3d2, l3_colors,0.2)
    # plt.show()
    plt.savefig('columns.png', dpi=1200)
    plt.clf()

