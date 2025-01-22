from pymilvus import connections, Collection
import numpy as np

class UDFGetEmbedding:
    def __init__(self):
        self.milvus_host = "localhost"
        self.milvus_port = 19530
        self.table_name = 'data_embedding'

    def transform(self, data, args, kvargs):
        print("enter UDFGetEmbedding success")
        if kvargs.get("host"):
            self.milvus_host = kvargs["host"].decode("utf-8")
        if kvargs.get("port"):
            self.milvus_port = int(kvargs["port"].decode("utf-8"))
        path_list_str = kvargs["nodes"].decode("utf-8")

        connections.connect(alias='default',host=self.milvus_host, port=self.milvus_port)
        collection = Collection(self.table_name)
        entities_iter = collection.query_iterator(
            expr="path in "+path_list_str,
            output_fields= ["path", "embedding"]
        )

        result = [["(path)", "(embedding)"], ['BINARY', 'BINARY']]
        while True:
            entities = entities_iter.next()
            if not entities:
                break
            for entity in entities:
                path = entity["path"]
                embedding = entity["embedding"]
                embedding_bytes = np.array(embedding, dtype="<f4").tobytes()
                result.append([path.encode('utf-8'), embedding_bytes])

        return result

if __name__ == '__main__':
    # 本地测试
    udf = UDFGetEmbedding()
    data = None
    args = None
    kvargs = {
        "host": "localhost".encode("utf-8"),
        "port": "19530".encode("utf-8"),
        "nodes": '["region", "nation", "part", "supplier", "customer", "orders", "lineitem"]'.encode("utf-8")
    }
    result = udf.transform(data, args, kvargs)
    print(result)
