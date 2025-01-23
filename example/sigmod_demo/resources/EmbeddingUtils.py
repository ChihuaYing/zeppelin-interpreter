from pymilvus import connections, Collection
import numpy as np
from sentence_transformers import SentenceTransformer

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

if __name__ == '__main__':
    # 本地测试
    udf = UDFSearchEmbedding()
    data = None
    args = None
    kvargs = {
        "host": "localhost".encode("utf-8"),
        "port": "19530".encode("utf-8"),
        "description": "订单".encode("utf-8")
    }
    result = udf.transform(data, args, kvargs)
    print(result)