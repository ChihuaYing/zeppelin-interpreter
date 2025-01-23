import json

from pymilvus import connections, Collection
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

class SentenceTransformerAccessor:
    def __init__(self, name_or_path="bert-base-chinese"):
        self._model_instance = SentenceTransformer(name_or_path)
        print("SentenceTransformerAccessor init success:", name_or_path)

    def encode(self, text):
        return self._model_instance.encode(text)

class MilvusDao:
    def __init__(self, kvargs):
        milvus_host = "localhost"
        milvus_port = 19530
        if kvargs.get("host"):
            milvus_host = kvargs["host"].decode("utf-8")
        if kvargs.get("port"):
            milvus_port = int(kvargs["port"].decode("utf-8"))

        table_name = 'data_embedding'
        connections.connect(alias='default', host=milvus_host, port=milvus_port)
        self.collection = Collection(table_name)

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

    def search_similarity(self, embedding):
        entities = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            output_fields=["path"],
            limit=10,
            param={"metric_type": "L2"},
        )

        return [ hit.fields["path"] for hit in entities[0] ]

    # TODO 后面需要 使用 cosine 相似度
    def bulk_search_similarity(self, embedding_list, path_list_json):
        entities = self.collection.search(
            data=embedding_list,
            anns_field="embedding",
            output_fields=["path"],
            limit=10,
            param={"metric_type": "L2"},
            expr="path in " + path_list_json
        )

        return [[(hit.fields["path"], hit.distance) for hit in entity] for entity in entities]


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

# if __name__ == '__main__':
#     # 本地测试
#     udf = UDFGetEmbedding()
#     data = None
#     args = None
#     kvargs = {
#         "host": "localhost".encode("utf-8"),
#         "port": "19530".encode("utf-8"),
#         "nodes": '["region", "nation", "part", "supplier", "customer", "orders", "lineitem"]'.encode("utf-8")
#     }
#     result = udf.transform(data, args, kvargs)
#     print(result)


class UDFSearchEmbedding:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFSearchEmbedding success")
        description = kvargs["description"].decode("utf-8")

        dao = MilvusDao(kvargs)
        model = SentenceTransformerAccessor()

        embedding = model.encode(description)
        paths = dao.search_similarity(embedding)

        result = [["(path)"], ['BINARY']]
        for path in paths:
            result.append([path.encode('utf-8')])

        return result

# if __name__ == '__main__':
#     # 本地测试
#     udf = UDFSearchEmbedding()
#     data = None
#     args = None
#     kvargs = {
#         "host": "localhost".encode("utf-8"),
#         "port": "19530".encode("utf-8"),
#         "description": "订单".encode("utf-8")
#     }
#     result = udf.transform(data, args, kvargs)
#     print(result)


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
        df = df[df['similarity'] > 100]

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