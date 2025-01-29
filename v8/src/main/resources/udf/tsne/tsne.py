import os
import pickle
import numpy as np

from pymilvus import connections, Collection
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class MilvusDao:
    def __init__(self, kvargs):
        milvus_host = "milvus"
        milvus_port = 19530
        if kvargs.get("host"):
            milvus_host = kvargs["host"].decode("utf-8")
        if kvargs.get("port"):
            milvus_port = int(kvargs["port"].decode("utf-8"))

        self.batch_size = 1000
        print("Connecting to Milvus")
        connections.connect(alias='default', host=milvus_host, port=milvus_port)
        self.collection = Collection(name='data_embedding')

    def fetch_all_embeddings(self):
        print("fetching all embeddings")
        iter = self.collection.query_iterator(
            output_fields=["path", "embedding"]
        )
        while True:
            entities = iter.next()
            if not entities:
                iter.close()
                break

            for entity in entities:
                yield entity["path"], entity["embedding"]


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

    def fit_tsne(self,paths:list[str], perplexity:int =30):
        filtered_embeddings = np.array([self.embeddings[path] for path in paths])
        tsne = TSNE(random_state=0, max_iter=5000, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(filtered_embeddings)
        return tsne_embeddings.T

    def visualize_2d(self, tsne_embeddings, s):
        # 需要坐标轴刻度，不需要坐标轴数字
        plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
        plt.scatter(tsne_embeddings[0], tsne_embeddings[1], s=s)
        plt.show()


if __name__ == '__main__':
    visualizer = TSNEVisualizer()
    print("Paths:", len(visualizer.embeddings))
    print("Fitting TSNE with 2 components for 1-level paths")
    l1_paths = [path for path in visualizer.embeddings if len(path.split(".")) == 1]
    for l1_perplexity in [50]:
        l1d2 = TSNEVisualizer.cache_get(f"l1d2_{l1_perplexity}", lambda: visualizer.fit_tsne(l1_paths, perplexity=l1_perplexity))
        visualizer.visualize_2d(l1d2, 3)
    print("Fitting TSNE with 2 components for 2-level paths")
    l2_paths = [path for path in visualizer.embeddings if len(path.split(".")) == 2]
    for l2_perplexity in [55]:
        l2d2 = TSNEVisualizer.cache_get(f"l2d2_{l2_perplexity}", lambda: visualizer.fit_tsne(l2_paths, perplexity=l2_perplexity))
        visualizer.visualize_2d(l2d2, 0.15)
    print("Fitting TSNE with 2 components for 3-level paths")
    l3_paths = [path for path in visualizer.embeddings if len(path.split(".")) == 3]
    for l3_perplexity in [55]:
        l3d2 = TSNEVisualizer.cache_get(f"l3d2_{l3_perplexity}", lambda: visualizer.fit_tsne(l3_paths, perplexity=l3_perplexity))
        visualizer.visualize_2d(l3d2, 0.005)

