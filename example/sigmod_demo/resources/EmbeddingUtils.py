import json
import shelve
from collections import defaultdict

from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
import numpy as np
from pymilvus.orm import utility
from sentence_transformers import SentenceTransformer
import pandas as pd
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math

class Encoder:
    def __init__(self):
        print("Initializing Sentence Transformer")
        self.encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
        print("Sentence Transformer Initialized")

    def encode(self, description):
        return self.encoder.encode(description)


class EmbeddingCalculator:
    def __init__(self):
        self.encoder = Encoder()
        print("Initializing Embedding Cache")
        self.embeddings= shelve.open('cache/embeddings', writeback=True)
        self.batch_size = 128
        self.child_weight = 0.3
        print("Embedding Calculator Initialized")

    def __enter__(self):
        self.embeddings.__enter__()
        return self

    def __exit__(self, type, value, trace):
        print("Closing... EmbeddingCalculator")
        self.embeddings.__exit__(type, value, trace)
        print("Closed")

    def calculate_leaf(self, leaf_paths: list[str]) -> list[np.array(np.float32)]:
        print("Calculating leaf embeddings for", len(leaf_paths), "paths")

        uncached_paths = [path for path in tqdm.tqdm(leaf_paths, desc="Finding uncached path") if path not in self.embeddings]
        print("Uncached paths:", len(uncached_paths))

        with tqdm.tqdm(total=len(leaf_paths), desc="Calculating embeddings") as pbar:
            for i in range(0, len(uncached_paths), self.batch_size):
                path_batch = uncached_paths[i:i + self.batch_size]
                description_batch = [self._get_path_description(path) for path in path_batch]
                embedding_batch = self.encoder.encode(description_batch)
                for path, embedding in zip(path_batch, embedding_batch):
                    self.embeddings[path] = embedding
                self.embeddings.sync()
                pbar.update(len(path_batch))

        return [self.embeddings[path] for path in tqdm.tqdm(leaf_paths, desc="Loading leaf path from cache")]

    def calculate(self, tree:dict[str, set], root:str) -> dict[str, np.array(np.float32)]:
        self.calculate_leaf(list(tree.keys()))

        cal_result = {}
        with tqdm.tqdm(total=len(tree), desc="Loading leaf path from cache and Calculating inner path embeddings") as pbar:
            self._load_and_calculate_embeddings(root, tree, cal_result, pbar)

        return cal_result

    def _load_and_calculate_embeddings(self, path:str, tree:dict[str, set], cal_result: dict[str, np.array(np.float32)], pbar):
        pbar.update(1)
        if not tree[path]:
            leaf_embedding = self.embeddings[path]
            cal_result[path] = leaf_embedding
            return leaf_embedding

        leaf_embeddings = self.embeddings[path]
        children_embeddings = [self._load_and_calculate_embeddings(child, tree, cal_result, pbar) for child in tree[path]]
        avg_children_embedding = np.mean(children_embeddings, axis=0)
        inner_embedding = leaf_embeddings * (1 - self.child_weight) + avg_children_embedding * self.child_weight
        cal_result[path] = inner_embedding
        return inner_embedding

    def _get_path_description(self, path:str) -> str:
        return " of ".join(path.split(".")[::-1])


class MilvusDao:
    def __init__(self, kvargs):
        milvus_host = "localhost"
        milvus_port = 19530
        if kvargs.get("host"):
            milvus_host = kvargs["host"].decode("utf-8")
        if kvargs.get("port"):
            milvus_port = int(kvargs["port"].decode("utf-8"))

        self.insert_batch_size = 1000
        connections.connect(alias='default', host=milvus_host, port=milvus_port)
        self.table_name = 'data_embedding'
        self.collection = self._create_or_load_collection(self.table_name)

    def _create_or_load_collection(self, table_name):
        if utility.has_collection(table_name):
            print(f"Collection {table_name} already exists, loading...")
            return Collection(name=table_name)

        print(f"Collection {table_name} does not exist, creating...")
        fields = [
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]

        schema = CollectionSchema(fields, description="embedding collection")
        collection = Collection(name=table_name, schema=schema)

        print("Creating index...")
        collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"})
        collection.create_index(field_name="path", index_params={"index_type": "Trie"})
        return collection

    def delete_all(self):
        print("Deleting all data in collection...")
        self.collection.drop()
        self.collection = self._create_or_load_collection(self.table_name)

    def insert_embeddings(self, embeddings):
        insert_results = []
        paths_to_insert = list(embeddings.keys())
        embeddings_to_insert = list(embeddings.values())
        with tqdm.tqdm(total=len(embeddings), desc="Inserting embeddings") as pbar:
            for i in range(0, len(embeddings), self.insert_batch_size):
                paths_batch = paths_to_insert[i:i + self.insert_batch_size]
                embeddings_batch = embeddings_to_insert[i:i + self.insert_batch_size]
                insert_result =self.collection.insert([paths_batch, embeddings_batch])
                insert_results.append(insert_result)
                pbar.update(len(paths_batch))
        return insert_results

    def load(self):
        self.collection.load()

    def search_similarity(self, embedding, path_list_json):
        entities = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            output_fields=["path"],
            limit=10,
            param={"metric_type": "COSINE"},
            expr="path in " + path_list_json
        )

        return [ hit.fields["path"] for hit in entities[0] ]

    def bulk_search_similarity(self, embedding_list, path_list_json):
        entities = self.collection.search(
            data=embedding_list,
            anns_field="embedding",
            output_fields=["path"],
            limit = 200,
            param={"metric_type": "COSINE"},
            expr="path in " + path_list_json
        )

        return [[(hit.fields["path"], hit.distance) for hit in entity] for entity in entities]


class UDFStoreEmbedding:
    def __init__(self):
        self.root_name = "root"

    def build_path_tree(self,paths: list[str]):
        path_tree = defaultdict(set)
        for path in paths:
            path_tree[path] = set()
            nodes = path.split(".")

            for i in range(len(nodes)):
                path_tree[".".join(nodes[:i])].add(".".join(nodes[:i+1]))

        path_tree[self.root_name] = path_tree.pop("")
        return path_tree

    def transform(self, data, args, kvargs):
        print("enter transform StoreEmbedding success")

        header = data[0]
        pathIndex = header.index('Path') if 'Path' in header else header.index('path')
        paths = [row[pathIndex].decode('utf-8') for row in data[2:]]

        print("start build path tree")
        paths_tree = self.build_path_tree(paths)
        print("finish build path tree")

        print("start calculate embeddings")
        with EmbeddingCalculator() as ec:
            embeddings = ec.calculate(paths_tree,self.root_name)
        print("finish calculate embeddings")

        print("start insert embeddings")
        dao = MilvusDao(kvargs)
        dao.delete_all()
        insert_results=dao.insert_embeddings(embeddings)
        print("finish insert embeddings:", insert_results)

        print("start load collection")
        dao.load()
        print("finish load collection")

        return [['(result)'], ['BINARY'], ["success".encode("utf-8")]]


if __name__ == '__main__':
    udf = UDFStoreEmbedding()
    paths = [
        "customer.c_custkey", "customer.c_name", "customer.c_address", "customer.c_nationkey", "customer.c_phone", "customer.c_acctbal", "customer.c_mktsegment", "customer.c_comment",
        "lineitem.l_orderkey", "lineitem.l_partkey", "lineitem.l_suppkey", "lineitem.l_linenumber", "lineitem.l_quantity", "lineitem.l_extendedprice", "lineitem.l_discount", "lineitem.l_tax", "lineitem.l_returnflag", "lineitem.l_linestatus", "lineitem.l_shipdate", "lineitem.l_commitdate", "lineitem.l_receiptdate", "lineitem.l_shipinstruct", "lineitem.l_shipmode", "lineitem.l_comment",
        "nation.n_nationkey", "nation.n_name", "nation.n_regionkey", "nation.n_comment",
        "orders.o_orderkey", "orders.o_custkey", "orders.o_orderstatus", "orders.o_totalprice", "orders.o_orderdate", "orders.o_orderpriority", "orders.o_clerk", "orders.o_shippriority", "orders.o_comment",
        "part.p_partkey", "part.p_name", "part.p_mfgr", "part.p_brand", "part.p_type", "part.p_size", "part.p_container", "part.p_retailprice", "part.p_comment",
        "partsupp.ps_partkey", "partsupp.ps_suppkey", "partsupp.ps_availqty", "partsupp.ps_supplycost", "partsupp.ps_comment",
        "region.r_regionkey", "region.r_name", "region.r_comment",
        "supplier.s_suppkey", "supplier.s_name", "supplier.s_address", "supplier.s_nationkey", "supplier.s_phone", "supplier.s_acctbal", "supplier.s_comment"
    ]
    data = [["path"],["BINARY"]] + [[path.encode("utf-8")] for path in paths]
    kvargs = {
        "host": "localhost".encode("utf-8"),
        "port": "19530".encode("utf-8"),
    }
    result = udf.transform(data, None, kvargs)
    print(result)


class UDFSearchEmbedding:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFSearchEmbedding success：", kvargs)
        description = kvargs["description"].decode("utf-8")
        paths_json = kvargs["paths"].decode("utf-8")

        dao = MilvusDao(kvargs)
        encoder = Encoder()

        embedding = encoder.encode(description)
        similar_paths = dao.search_similarity(embedding, paths_json)

        result = [["(path)"], ['BINARY']]
        for path in similar_paths:
            result.append([path.encode('utf-8')])

        return result


if __name__ == '__main__':
    # 本地测试
    udf = UDFSearchEmbedding()
    data = None
    args = None
    kvargs = {
        "host": "localhost".encode("utf-8"),
        "port": "19530".encode("utf-8"),
        "description": "订单".encode("utf-8"),
        "paths": '["region.r_regionkey", "region.r_name", "region.r_comment", "supplier.s_phone","orders"]'.encode("utf-8")
    }
    result = udf.transform(data, args, kvargs)
    print(result)


class UDFAnalyseRelation:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFAnalyseRelation success")
        sources_json = kvargs["sources"].decode("utf-8")
        targets_json = kvargs["targets"].decode("utf-8")

        source_paths = json.loads(sources_json)
        print("Calculating embeddings for sources:", len(source_paths))

        with EmbeddingCalculator() as ec:
            source_embeddings = ec.calculate_leaf(source_paths)

        dao = MilvusDao(kvargs)

        search_result = dao.bulk_search_similarity(source_embeddings, targets_json)
        relations = [
            (source_path, target_path, similarity)
            for i, source_path in enumerate(source_paths)
            for target_path, similarity in search_result[i]
        ]
        # Create a DataFrame from the relations list with columns: source, target, and similarity
        df = pd.DataFrame(relations, columns=["source", "target", "similarity"])

        # Filter out records with similarity less than or equal to 100
        df = df[df['similarity'] > 0.5]

        # Extract the root part (before the first dot) from source and target columns
        df['source_root'] = df['source'].str.split('.').str[0]
        df['target_root'] = df['target'].str.split('.').str[0]

        # Filter out records where source_root and target_root are the same
        df = df[df['source_root'] != df['target_root']]

        # Find the maximum similarity for each unique pair of source_root and target_root
        max_similarities = df.loc[df.groupby([df[['source_root', 'target_root']].apply(frozenset, axis=1)])['similarity'].idxmax()]

        # Select only the source, target, and similarity columns from the DataFrame
        relations = max_similarities[['source', 'target', 'similarity']]

        # Encode the source and target columns as UTF-8
        relations.loc[:, 'source'] = relations['source'].str.encode('utf-8')
        relations.loc[:, 'target'] = relations['target'].str.encode('utf-8')

        # Return the final result as a list of lists, including headers and data types
        return [["(source)", "(target)", "(score)"], ['BINARY', 'BINARY', 'DOUBLE']] + relations.values.tolist()

if __name__ == '__main__':
    # 本地测试
    udf = UDFAnalyseRelation()
    data = None
    args = None
    kvargs = {
        "host": "localhost".encode("utf-8"),
        "port": "19530".encode("utf-8"),
        "sources": '["region.r_regionkey", "region.r_name", "region.r_comment", "supplier.s_phone"]'.encode("utf-8"),
        "targets": '["supplier.s_suppkey", "supplier.s_name", "supplier.s_address", "nation.n_name", "nation.n_comment", "part", "customer", "orders", "lineitem"]'.encode("utf-8")
    }
    result = udf.transform(data, args, kvargs)
    print(result)

class Node:
    def __init__(self, path='', embedding=None):
        self.path = path
        self.embedding = embedding if embedding is not None and embedding.size > 0 else np.array([])


class Aggregator:
    def __init__(self, threshold=0.8, n_clusters=8, pca_components=50, tsne_components=3):
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.tsne_components = tsne_components

    def compute_similarity_matrix(self, embeddings):
        # embeddings = [node.embedding for node in nodes]
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def calculate_silhouette_score(self, similarity_matrix, labels):
        """
        计算聚类的轮廓系数。
        """
        distance_matrix = 1 - similarity_matrix
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.maximum(0, distance_matrix)
        try:
            silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')
            return silhouette_avg
        except ValueError:
            return -1.0

    def calculate_avg_similarity(self, nodes, labels):
        """
        计算聚类内的平均相似度。
        """
        avg_similarity = 0
        total_pairs = 0
        for label in np.unique(labels):  # 遍历每个簇
            indices = np.where(labels == label)[0]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    node_i = nodes[indices[i]]
                    node_j = nodes[indices[j]]
                    similarity = cosine_similarity([node_i.embedding], [node_j.embedding])[0][0]
                    avg_similarity += similarity
                    total_pairs += 1
        if total_pairs > 0:
            return avg_similarity / total_pairs
        return 0

    def calculate_cluster_separation(self, similarity_matrix, labels):
        cluster_separation = 0
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                cluster_i = np.where(labels == unique_labels[i])[0]
                cluster_j = np.where(labels == unique_labels[j])[0]
                # 类间的平均距离
                inter_cluster_distance = np.mean([1 - similarity_matrix[x][y] for x in cluster_i for y in cluster_j])
                cluster_separation += inter_cluster_distance
        return cluster_separation / (num_clusters * (num_clusters - 1) / 2 if num_clusters > 1 else 1)

    def find_optimal_clusters(self, similarity_matrix, max_clusters=5):
        """
        自动选择最优的聚类数，根据轮廓系数。
        返回最优聚类数、对应的 labels 和最大轮廓系数。
        """
        best_score = -1
        best_n_clusters = self.n_clusters
        best_labels = None

        for n_clusters in range(3, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(similarity_matrix)
            labels = kmeans.labels_
            silhouette_avg = self.calculate_silhouette_score(similarity_matrix, labels)
            print(f"聚类数为：{n_clusters}  轮廓系数为：{silhouette_avg}")

            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_n_clusters = n_clusters
                best_labels = labels

        return best_n_clusters, best_labels, best_score

    def apply_pca(self, embeddings, n_samples):
        n_components = min(self.pca_components, n_samples)
        pca = PCA(n_components=n_components)
        pca_embeddings = pca.fit_transform(embeddings)
        return pca_embeddings

    def apply_tsne(self, embeddings, n_samples):
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=self.tsne_components, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(embeddings)
        return tsne_embeddings

    # 聚合
    def aggregate(self, root_nodes):
        embeddings = [node.embedding for node in root_nodes]
        n_samples = len(root_nodes)

        # 先进行 PCA 降维
        pca_embeddings = self.apply_pca(embeddings, n_samples)
        print(f"PCA 降维后的形状：{pca_embeddings.shape}")

        # 再进行 t-SNE 降维
        tsne_embeddings = self.apply_tsne(pca_embeddings, n_samples)
        print(f"t-SNE 降维后的形状：{tsne_embeddings.shape}")

        similarity_matrix = self.compute_similarity_matrix(tsne_embeddings)
        print(f"相似度矩阵为：{similarity_matrix}")
        print(f"原先森林数为: {len(root_nodes)}")

        optimal_n_clusters, labels, silhouette_avg = self.find_optimal_clusters(
            similarity_matrix=similarity_matrix,
            max_clusters=max(int(math.sqrt(len(root_nodes)) * 1.5), 3)
        )
        print(f"最优聚类数: {optimal_n_clusters}")
        print(f"轮廓系数: {silhouette_avg}")

        avg_similarity = self.calculate_avg_similarity(root_nodes, labels)
        cluster_separation = self.calculate_cluster_separation(similarity_matrix, labels)

        buckets = {}
        for idx, label in enumerate(labels):
            if label not in buckets:
                buckets[label] = []
            buckets[label].append(root_nodes[idx])

        node_components = list(buckets.values())
        print(f"分块结果：{[[node.path for node in component] for component in node_components]}")

        return silhouette_avg, avg_similarity, cluster_separation, labels


class UDFMerge:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter transform Merge success")
        path_list = json.loads(kvargs["str"].decode("utf-8"))

        with EmbeddingCalculator() as ec:
            embeddings = ec.calculate_leaf(path_list)

        nodes = [Node(path=path, embedding=embedding) for path, embedding in zip(path_list, embeddings)]

        aggregator = Aggregator()
        silhouette_avg, avg_similarity, cluster_separation, labels = aggregator.aggregate(nodes)
        print("finish aggregate")
        print(f"silhouette_avg为: {silhouette_avg}\navg_similarity为: {avg_similarity}\ncluster_separation为: {cluster_separation}")

        # 生成最终的结果
        result = self.generate_result(nodes, labels)
        print("finish generate result")

        return result

    def generate_result(self, root_nodes, labels):
        result = [['(name)', '(label)'], ['BINARY', 'BINARY']]
        for idx, node in enumerate(root_nodes):
            result.append([node.path.encode('utf-8'), str(labels[idx]).encode('utf-8')])
        return result