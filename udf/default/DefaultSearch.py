import numpy as np

from api.BaseSearch import UDFBaseSearch
from default.DefaultUtilities import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION
from pymilvus import connections, Collection

class UDFDefaultSearch(UDFBaseSearch):
    def __init__(self, milvus_host=MILVUS_HOST, milvus_port=MILVUS_PORT, collection_name = MILVUS_COLLECTION):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name

    def search(self, pattern: str, embedding: np.ndarray, limit:int) -> list[tuple[str, str, str, float]]:
        print(f"fetching pattern: {pattern}, limit: {limit}")

        milvus_pattern = pattern.replace("*", "%")
        expr = 'path like "' + milvus_pattern + '"'

        connections.connect(host=self.milvus_host, port=self.milvus_port)
        collection = Collection(self.collection_name)

        entities = collection.search(
            data=[embedding],
            anns_field="embedding",
            output_fields=["path","type","description"],
            limit=limit,
            param={"metric_type": "COSINE"},
            expr=expr,
            search_params={
                "hints": "iterative_filter"
            }
        )

        return [
            (
                hit.fields["path"],
                hit.fields["type"] if "type" in hit.fields else None,
                hit.fields["description"],
                hit.distance
            )
            for hit in entities[0]
        ]

    def describe_similarity(self, source_description: str, target_descriptions: list[str]) -> list[str]:
        return [
            f"{source_description} is similar to {target_description}"
            for target_description in target_descriptions
        ]

if __name__ == "__main__":
    from default.DefaultEncode import UDFDefaultEncode

    encoder = UDFDefaultEncode(cache_path="embeddings")
    encoder_data = [ ['description'], ['BINARY'] ,["test description".encode()]]
    encoder_result = encoder.transform(encoder_data, [], {})
    print(encoder_result)

    encoder_result[0] = [name[1:-1] for name in encoder_result[0]]

    searcher = UDFDefaultSearch()
    searched_data = encoder_result
    searched_result = searcher.transform(searched_data, [], {
        "pattern": "*".encode(),
        "limit": 10,
    })
    print(searched_result)
