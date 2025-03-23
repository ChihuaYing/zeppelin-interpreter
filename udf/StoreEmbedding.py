import json
import shelve
import tqdm
import numpy as np

from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from pymilvus.orm import utility
from sentence_transformers import SentenceTransformer

class Encoder:
    def __init__(self, cache_path='cache/embeddings'):
        print("Initializing Sentence Transformer")
        self.encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.batch_size = 128
        self.cache_path = cache_path
        self.child_weight = 0.3

    def encode(self, descriptions: list[str]) -> list[np.array]:
        print("Calculating embeddings of", len(descriptions), "descriptions")

        # 打开 shelve 缓存
        with shelve.open(self.cache_path) as cache:
            # 找出未缓存的描述
            uncached_descriptions = [desc for desc in descriptions if desc not in cache]
            print("Uncached descriptions:", len(uncached_descriptions))

            # 计算未缓存的 embeddings
            if uncached_descriptions:
                description_batches = [
                    uncached_descriptions[i:i + self.batch_size]
                    for i in range(0, len(uncached_descriptions), self.batch_size)
                ]

                with tqdm.tqdm(total=len(uncached_descriptions), desc="Calculating embeddings") as pbar:
                    for description_batch in description_batches:
                        embedding_batch = self.encoder.encode(description_batch)
                        for desc, embedding in zip(description_batch, embedding_batch):
                            cache[desc] = embedding  # 存入缓存
                        cache.sync()  # 立即写入
                        pbar.update(len(description_batch))

            # 读取缓存
            embeddings = [cache[desc] for desc in descriptions]

        return embeddings


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
        self.collection = self._create_or_load_collection()

    def _create_or_load_collection(self):
        table_name = 'embeddings'

        if utility.has_collection(table_name):
            print(f"Loading collection {table_name}")
            return Collection(table_name)

        print(f"Creating collection {table_name}")
        fields = [
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4097)
        ]

        schema = CollectionSchema(fields, description="embedding collection")
        collection = Collection(table_name, schema=schema)

        print("Creating index...")
        collection.create_index(field_name="embedding",
                                index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "nlist": 1024})
        collection.create_index(field_name="path", index_params={"index_type": "Trie"})
        return collection

    def delete_all(self):
        print("Recreating collection")
        self.collection.drop()
        self.collection = self._create_or_load_collection()

    def insert_embeddings(self, paths, embeddings, descriptions):
        assert len(paths) == len(embeddings)
        assert len(paths) == len(descriptions)
        insert_results = []
        with tqdm.tqdm(total=len(paths), desc="Inserting embeddings") as pbar:
            for i in range(0, len(paths), self.batch_size):
                paths_batch = paths[i:i + self.batch_size]
                embeddings_batch = embeddings[i:i + self.batch_size]
                descriptions_batch = descriptions[i:i + self.batch_size]
                insert_result = self.collection.insert([paths_batch, embeddings_batch, descriptions_batch])
                insert_results.append(insert_result)
                pbar.update(len(paths_batch))
        return insert_results

    def load(self):
        print("Loading milvus collection")
        self.collection.load()


class UDFStoreEmbedding:
    def __init__(self):
        pass

    def _build_descriptions(self, paths: list[str]) -> dict[str, str]:
        print("building descriptions for", len(paths), "paths")
        nodes_list = [path.split(".") for path in paths]
        tree = self._build_path_tree(nodes_list)

        descriptions = {}
        self._dfs_build_descriptions(tree, [], descriptions)
        return descriptions

    def _build_path_tree(self, nodes_list: list[list[str]]) -> dict:
        print("building path tree for", len(nodes_list), "paths")

        tree = {}
        for nodes in nodes_list:
            current_node = tree
            for node in nodes:
                current_node = current_node.setdefault(node, {})
        return tree

    def _dfs_build_descriptions(self, tree, path_nodes: list[str], descriptions: dict[str, str]) -> None:
        for key, value in tree.items():
            path_nodes.append(key)
            description_nodes = list(reversed(path_nodes))
            if not value:
                description_nodes += list(tree.keys())
            else:
                description_nodes += list(value.keys())
            descriptions[".".join(path_nodes)] = ", ".join(description_nodes)
            self._dfs_build_descriptions(value, path_nodes, descriptions)
            path_nodes.pop()

    def transform(self, data, args, kvargs):
        print("enter UDF StoreEmbedding")

        header = data[0]
        pathIndex = header.index('Path') if 'Path' in header else header.index('path')
        paths = [row[pathIndex].decode('utf-8') for row in data[2:]]

        description_each_path = self._build_descriptions(paths)
        paths_contain_inner_node = list(description_each_path.keys())
        descriptions = list(description_each_path.values())
        ec = Encoder()  # 直接创建实例
        embeddings = ec.encode(descriptions)

        dao = MilvusDao(kvargs)
        dao.delete_all()

        insert_results = dao.insert_embeddings(paths_contain_inner_node, embeddings, descriptions)
        success_count = sum([insert_result.succ_count for insert_result in insert_results])
        err_count = sum([insert_result.err_count for insert_result in insert_results])

        dao.load()

        return [
            ["(inserted)", "(failed)"],
            ["LONG", "LONG"],
            [success_count, err_count]
        ]


# if __name__ == '__main__':
#     udf = UDFStoreEmbedding()
#     paths = [
#         "customer.c_custkey", "customer.c_name", "customer.c_address", "customer.c_nationkey", "customer.c_phone", "customer.c_acctbal", "customer.c_mktsegment", "customer.c_comment",
#         "lineitem.l_orderkey", "lineitem.l_partkey", "lineitem.l_suppkey", "lineitem.l_linenumber", "lineitem.l_quantity", "lineitem.l_extendedprice", "lineitem.l_discount", "lineitem.l_tax", "lineitem.l_returnflag", "lineitem.l_linestatus", "lineitem.l_shipdate", "lineitem.l_commitdate", "lineitem.l_receiptdate", "lineitem.l_shipinstruct", "lineitem.l_shipmode", "lineitem.l_comment",
#         "nation.n_nationkey", "nation.n_name", "nation.n_regionkey", "nation.n_comment",
#         "orders.o_orderkey", "orders.o_custkey", "orders.o_orderstatus", "orders.o_totalprice", "orders.o_orderdate", "orders.o_orderpriority", "orders.o_clerk", "orders.o_shippriority", "orders.o_comment",
#         "part.p_partkey", "part.p_name", "part.p_mfgr", "part.p_brand", "part.p_type", "part.p_size", "part.p_container", "part.p_retailprice", "part.p_comment",
#         "partsupp.ps_partkey", "partsupp.ps_suppkey", "partsupp.ps_availqty", "partsupp.ps_supplycost", "partsupp.ps_comment",
#         "region.r_regionkey", "region.r_name", "region.r_comment",
#         "supplier.s_suppkey", "supplier.s_name", "supplier.s_address", "supplier.s_nationkey", "supplier.s_phone", "supplier.s_acctbal", "supplier.s_comment"
#     ]
#     data = [["path"],["BINARY"]] + [[path.encode("utf-8")] for path in paths]
#     kvargs = {
#         "host": "localhost".encode("utf-8"),
#         "port": "19530".encode("utf-8"),
#     }
#     result = udf.transform(data, None, kvargs)
#     print(result)
