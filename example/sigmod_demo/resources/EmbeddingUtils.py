import shelve
from collections import defaultdict

from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
import numpy as np
from pymilvus.orm import utility
from sentence_transformers import SentenceTransformer
import pandas as pd
import tqdm


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

    def calculate(self, tree:dict[str, set], root:str) -> dict[str, np.array(np.float32)]:
        print("Calculating embeddings for", len(tree), "paths")

        uncached_paths = [path for path in tqdm.tqdm(tree.keys(), desc="Finding uncached path") if path not in self.embeddings]
        print("Uncached paths:", len(uncached_paths))

        with tqdm.tqdm(total=len(uncached_paths), desc="Calculating uncached paths embeddings") as pbar:
            for i in range(0, len(uncached_paths), self.batch_size):
                path_batch = uncached_paths[i:i + self.batch_size]
                description_batch = [self._get_path_description(path) for path in path_batch]
                embedding_batch = self.encoder.encode(description_batch)
                for path, embedding in zip(path_batch, embedding_batch):
                    self.embeddings[path] = embedding
                self.embeddings.sync()
                pbar.update(len(path_batch))

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

    def get_embedding(self, path_list_json):
        entities_iter = self.collection.query_iterator(
            expr="path in " + path_list_json,
            output_fields=["path", "embedding"]
        )

        embedding_each_path = {}
        while True:
            entities = entities_iter.next()
            if not entities:
                break
            for entity in entities:
                path = entity["path"]
                embedding = entity["embedding"]
                embedding_each_path[path] = embedding

        return embedding_each_path

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


class UDFGetEmbedding:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFGetEmbedding success")
        path_list_json = kvargs["nodes"].decode("utf-8")

        dao = MilvusDao(kvargs)

        embedding_each_path = dao.get_embedding(path_list_json)

        result = [["(path)", "(embedding)"], ['BINARY', 'BINARY']]
        for path, embedding in embedding_each_path.items():
            embedding_bytes = np.array(embedding, dtype="<f4").tobytes()
            result.append([path.encode('utf-8'), embedding_bytes])
        return result


if __name__ == '__main__':
    # 本地测试
    udf = UDFGetEmbedding()
    kvargs = {
        "host": "localhost".encode("utf-8"),
        "port": "19530".encode("utf-8"),
        "nodes": '["region", "nation", "part", "supplier", "customer", "orders", "lineitem"]'.encode("utf-8")
    }
    result = udf.transform(None, None, kvargs)
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

        dao = MilvusDao(kvargs)
        source_embeddings = dao.get_embedding(sources_json)
        source_paths = list(source_embeddings.keys())
        source_embeddings = list(source_embeddings.values())
        result = dao.bulk_search_similarity(source_embeddings, targets_json)
        relations = [
            (source_path, target_path, similarity)
            for i, source_path in enumerate(source_paths)
            for target_path, similarity in result[i]
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