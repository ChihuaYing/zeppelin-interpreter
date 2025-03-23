from numpy import typing as npt
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from pymilvus.orm import utility
import tqdm

from BaseInserter import UDFBaseInserter

class UDFDefaultInserter(UDFBaseInserter):

    def __init__(self, milvus_host="milvus", milvus_port=19530, collection_name = "embeddings", batch_size=1000):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.batch_size = batch_size

    def _recreate_and_load_collection(self):
        print(f"Recreating collection {self.collection_name} of Milvus in {self.milvus_host}:{self.milvus_port}")
        connections.connect(host=self.milvus_host, port=self.milvus_port)

        if utility.has_collection(self.collection_name):
            print(f"Collection already exists, dropping it")
            Collection(self.collection_name).drop()

        print(f"Creating collection")
        collection = Collection(self.collection_name, schema=CollectionSchema([
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=10,nullable=True),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4097),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]))

        print("Creating index")
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "nlist": 1024,
            },
        )
        collection.create_index(
            field_name="path",
            index_params={
                "index_type": "Trie",
            },
        )
        collection.load()
        return collection

    def insert(self,paths: list[str],types: list[str],descriptions: list[str], embeddings: list[npt.NDArray]) -> tuple[int, int]:
        insert_results = []
        collection = self._recreate_and_load_collection()
        with tqdm.tqdm(total=len(paths), desc="Inserting embeddings") as pbar:
            for i in range(0, len(paths), self.batch_size):
                paths_batch = paths[i:i + self.batch_size]
                types_batch = types[i:i + self.batch_size]
                descriptions_batch = descriptions[i:i + self.batch_size]
                embeddings_batch = embeddings[i:i + self.batch_size]
                insert_result = collection.insert([paths_batch, types_batch, descriptions_batch, embeddings_batch])
                insert_results.append(insert_result)
                pbar.update(len(paths_batch))
        inserted = sum([result.insert_count for result in insert_results])
        failed = len(paths) - inserted
        return inserted, failed

if __name__ == "__main__":
    from DefaultDescriptor import UDFDefaultDescriptor
    descriptor = UDFDefaultDescriptor()
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
    descriptor_data = [['path', 'type'], ['BINARY', 'BINARY']] + [[path.encode(), "BINARY".encode()] for path in paths]
    descriptor_result = descriptor.transform(descriptor_data, [], {})
    print(descriptor_result)

    descriptor_result[0] = [ name[1:-1] for name in descriptor_result[0]]

    from  DefaultEncoder import UDFDefaultEncoder
    encoder = UDFDefaultEncoder(cache_path="embeddings")
    encoder_data = descriptor_result
    encoder_result = encoder.transform(encoder_data, [], {})
    print(encoder_result)

    encoder_result[0] = [name[1:-1] for name in encoder_result[0]]

    inserter = UDFDefaultInserter(milvus_host="127.0.0.1")
    inserter_data = encoder_result
    inserter_result = inserter.transform(inserter_data, [], {})
    print(inserter_result)
