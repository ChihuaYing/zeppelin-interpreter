from abc import abstractmethod

import numpy as np

class MilvusDao:
    def __init__(self, kvargs):
        milvus_host = "127.0.0.1"
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
                                index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"})
        collection.create_index(field_name="path", index_params={"index_type": "Trie"})
        return collection

    def load(self):
        print("Loading milvus collection")
        self.collection.load()

    def fetch_first_level(self, pattern: str):
        print(f"fetching first level with pattern: {pattern}")
        sub_pattern = pattern.split('.')[0].replace('*', '%')
        entities = self.collection.query(
            output_fields=["path", "embedding", "description"],
            expr=f"path like '{sub_pattern}' and not path like '%.%'"
        )
        return [(entity["path"], entity["embedding"], entity["description"]) for entity in entities]


class UDFBaseFetch:
    @abstractmethod
    def fetch(self, pattern: str, level:int) -> list[tuple[str,str,str,np.ndarray]]:
        pass

    def transform(self, data, args, kvargs):
        print(f"enter {self.__class__.__name__}")
        print(f"data[0]: {data[0]}")
        print(f"data[1]: {data[1]}")
        print(f"len(data): {len(data)}")
        print(f"args: {args}")
        print(f"kvargs: {kvargs}")

        pattern = kvargs["pattern"].decode("utf-8")
        level = int(kvargs["level"])
        result = self.fetch(pattern,level)

        return [
            ['(path)','(type)','(description)','(embedding)'],
            ['BINARY','BINARY','BINARY','BINARY'],
        ] + [
            [path.encode(), type.encode() if type else None, description.encode(), embedding.tobytes()]
            for path, type, description, embedding in result
        ]
