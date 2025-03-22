import asyncio
import json
from collections import defaultdict
from typing import NamedTuple
from tenacity import retry, stop_after_attempt, wait_incrementing

import aiohttp
import shelve
import tqdm
import numpy as np
import pandas as pd

from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from pymilvus.orm import utility
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class LLMDao:
    def __init__(self):
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.key = "204a3ea9bf39f18dd9bf32c71ecbb607.mITgz6pgV7Hzj27A"
        self.model = "GLM-4-Flash"


    @retry(stop=stop_after_attempt(6), wait=wait_incrementing(start=1, increment=1))
    async def _fetch(self, session, prompt: str, semaphore: asyncio.Semaphore, pbar: tqdm.tqdm) -> str:
        async with semaphore:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}"
            }
            request_body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "do_sample": False,
            }
            async with session.post(self.url, headers=headers, json=request_body) as response:
                response.raise_for_status()
                response_data = await response.json()
                choices = response_data.get("choices", [])
                if not choices:
                    return "Error: Missing or empty 'choices' in response"
                pbar.update(1)
                return choices[0].get("message", {}).get("content", "")

    async def _bulk_fetch(self, prompts: list[str], max_concurrent_requests) -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        async with aiohttp.ClientSession() as session:
            with tqdm.tqdm(total=len(prompts), desc="Requesting to LLM") as pbar:
                tasks = [self._fetch(session, prompt, semaphore, pbar) for prompt in prompts]
            return await asyncio.gather(*tasks)

    def _bulk_get_response(self, prompts: list[str], max_concurrent_requests: int = 20) -> list[str]:
        return asyncio.run(self._bulk_fetch(prompts, max_concurrent_requests))

    def provide_keywords(self, decription: str):
        prompt = f"""
        Below is a description intended for searching specific information. 
        Please provide a set of relevant keywords from the description to improve the search results, 
        separated by commas and ranked by relevance. Ensure to output only the keywords without any additional text.
        Description: {decription}
        """
        response = self._bulk_get_response([prompt])[0]
        return response

    def describe_relation(self, pairs: list[tuple[str, str]]) -> list[str]:
        prompts = [
            f"""
            You are a concept master. 
            Please provide a concise and specific description of the relationship 
            between "{pair[0]}" and "{pair[1]}" in one sentence using no more than 10 words and without any punctuation. 
            Only the summary result is needed. Ensure the summary does not exceed 10 words and is concise and logical phrase.
            """
            for pair in pairs
        ]
        return self._bulk_get_response(prompts)

    def summarize(self, path_clusters: list[list[str]]) -> list[str]:
        prompts = [
            """
            You are a summarization expert. 
            I will provide several lines of phrases delimited by commas.
            Please summarize them into one sentence using no more than 10 words characters and without any punctuation. 
            Only the summary result is needed. Ensure the summary does not exceed 10 words and is concise and logical phrase.
            
            Here are the phrases to summarize:
            {}
            """.format("\n".join(path_cluster))
            for path_cluster in path_clusters
        ]
        return self._bulk_get_response(prompts)


# if __name__ == '__main__':
#     # 本地测试
#     keywords = LLMDao().summarize([
#         ['customer.c_acctbal', 'orders.o_comment', 'supplier.s_acctbal'],
#         ['customer.c_address', 'lineitem.l_returnflag'],
#         ['customer.c_comment', 'customer.c_mktsegment', 'nation.n_regionkey'],
#         ['customer.c_custkey', 'customer.c_name', 'nation.n_comment', 'nation.n_nationkey', 'orders.o_orderstatus', 'supplier.s_suppkey'],
#         ['customer.c_nationkey', 'orders.o_clerk', 'orders.o_shippriority'],
#         ['customer.c_phone', 'lineitem.l_commitdate', 'region.r_comment', 'region.r_regionkey'],
#         ['lineitem.l_comment', 'nation.n_name', 'region.r_name', 'supplier.s_comment', 'supplier.s_nationkey'],
#         ['lineitem.l_discount', 'orders.o_orderkey', 'part.p_comment', 'part.p_type'],
#         ['lineitem.l_extendedprice', 'lineitem.l_linestatus', 'lineitem.l_suppkey'],
#         ['lineitem.l_linenumber', 'orders.o_orderdate', 'orders.o_orderpriority', 'supplier.s_phone'],
#         ['lineitem.l_orderkey', 'lineitem.l_quantity'],
#         ['lineitem.l_partkey', 'orders.o_custkey', 'orders.o_totalprice'],
#         ['lineitem.l_receiptdate', 'lineitem.l_tax'],
#         ['lineitem.l_shipdate', 'lineitem.l_shipmode', 'supplier.s_address'],
#         ['lineitem.l_shipinstruct', 'part.p_retailprice', 'partsupp.ps_availqty', 'partsupp.ps_suppkey'],
#         ['part.p_brand', 'part.p_container', 'part.p_name', 'part.p_size', 'supplier.s_name'],
#         ['part.p_mfgr', 'part.p_partkey'],
#         ['partsupp.ps_comment', 'partsupp.ps_partkey', 'partsupp.ps_supplycost'],
#     ])
#     print(keywords)


class Encoder:
    def __init__(self):
        print("Initializing Sentence Transformer")
        self.encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
        print("Initializing Embedding Cache")
        self.embeddings = shelve.open('cache/embeddings', writeback=True)
        self.batch_size = 128
        self.child_weight = 0.3

    def __enter__(self):
        self.embeddings.__enter__()
        return self

    def __exit__(self, type, value, trace):
        print("Closing... EmbeddingCalculator")
        self.embeddings.__exit__(type, value, trace)
        print("Closed")

    def encode(self, descriptions: list[str]) -> list[np.array(np.float32)]:
        print("Calculating embeddings of", len(descriptions), "descriptions")

        uncached_descriptions = [
            description
            for description in tqdm.tqdm(descriptions, desc="Finding uncached descriptions")
            if description not in self.embeddings
        ]
        print("Uncached descriptions:", len(uncached_descriptions))

        description_batches = [
            uncached_descriptions[i:i + self.batch_size]
            for i in range(0, len(uncached_descriptions), self.batch_size)
        ]
        with tqdm.tqdm(total=len(uncached_descriptions), desc="Calculating embeddings") as pbar:
            for description_batch in description_batches:
                embedding_batch = self.encoder.encode(description_batch)
                for description, embedding in zip(description_batch, embedding_batch):
                    self.embeddings[description] = embedding
                self.embeddings.sync()
                pbar.update(len(description_batch))

        return [
            self.embeddings[description]
            for description in tqdm.tqdm(descriptions, desc="Gathering embeddings from cache")
        ]


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
        # connections.connect(uri=f"cache/milvus.db")
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

    def search_similarity(self, embedding, path_list_json):
        print("Searching similar embeddings")

        entities = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            output_fields=["path"],
            limit=10,
            param={"metric_type": "COSINE"},
            expr="path in " + path_list_json
        )

        return [hit.fields["path"] for hit in entities[0]]


    def fetch_by_path(self, paths_json: str):
        print("fetching by path")
        paths_list = json.loads(paths_json)
        expr = f"path in {paths_list}"
        entities = self.collection.query(
            output_fields=["path", "embedding", "description"],
            expr=expr
        )
        return [(entity["path"], entity["embedding"], entity["description"]) for entity in entities]

    def bulk_search_similarity(self, embedding_list, path_list_json):
        print("Searching similar embeddings")
        path_list = json.loads(path_list_json)
        expr = f"path in {path_list}"
        print("Generated Milvus query expression:", expr)
        entities = self.collection.search(
            expr=expr,
            data=embedding_list,
            anns_field="embedding",
            output_fields=["path", "description"],
            limit=10,
            param={"metric_type": "COSINE"},
        )

        return [
            [(hit.fields["path"], hit.fields["description"], hit.distance) for hit in entity]
            for entity in entities
        ]



class UDFAnalyseRelation:
    RESULT_HEADER = [
        ["(source)", "(target)", "(score)", "(description)"],
        ['BINARY', 'BINARY', 'DOUBLE', 'BINARY'],
    ]

    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFAnalyseRelation success:", kvargs)
        sources_json = kvargs["sources"].decode("utf-8")
        targets_json = kvargs["targets"].decode("utf-8")

        dao = MilvusDao(kvargs)
        source_result = list(dao.fetch_by_path(sources_json))
        print("source_result:", len(source_result))

        if not source_result:
            return self.RESULT_HEADER

        source_paths, source_embeddings, source_descriptions = (list(t) for t in zip(*source_result))

        target_result = dao.bulk_search_similarity(source_embeddings, targets_json)
        relations = [
            (source_path, target_path, similarity, source_description, target_description)
            for i, (source_path, source_description) in enumerate(zip(source_paths, source_descriptions))
            for target_path, target_description, similarity in target_result[i]
        ]

        print("relations:", len(relations))

        if not relations:
            return self.RESULT_HEADER

        # Create a DataFrame from the relations list with columns: source, target, and similarity
        df = pd.DataFrame(relations,
                          columns=["source", "target", "similarity", "source_description", "target_description"])

        # Extract the root part (before the first dot) from source and target columns
        df['source_root'] = df['source'].str.split('.').str[0]
        df['target_root'] = df['target'].str.split('.').str[0]

        # Filter out records where source_root and target_root are the same
        df = df[df['source_root'] != df['target_root']]

        # Find the maximum similarity for each unique pair of source_root and target_root
        max_similarities = df.loc[
            df.groupby([df[['source_root', 'target_root']].apply(frozenset, axis=1)])['similarity'].idxmax()]

        # Select only the source, target, and similarity columns from the DataFrame
        relations = max_similarities[['source', 'target', 'similarity', 'source_description', 'target_description']]

        # Select top 1/3 of the most relations based on similarity
        limit = 3
        relations = relations.sort_values(by='similarity', ascending=False).head(limit)

        if len(relations) == 0:
            return self.RESULT_HEADER

        # 并行获取 source_description 和 target_description 的关系的描述
        relations['description'] = LLMDao().describe_relation(
            list(zip(relations['source_description'], relations['target_description'])))
        relations = relations[['source', 'target', 'similarity', 'description']]

        # Encode the source and target columns as UTF-8
        relations.loc[:, 'source'] = relations['source'].str.encode('utf-8')
        relations.loc[:, 'target'] = relations['target'].str.encode('utf-8')
        relations.loc[:, 'description'] = relations['description'].str.encode('utf-8')

        # Return the final result as a list of lists, including headers and data types
        return self.RESULT_HEADER + relations.values.tolist()


# if __name__ == '__main__':
#     # 本地测试
#     udf = UDFAnalyseRelation()
#     kvargs = {'sources': b'["1612SMShuvo_sub_contract","2010USFJava_MaurerP","1lirisist_kanbank"]', 'port': b'19530', 'targets': b'["1lirisist_kanbank","1ibrary_1ibrary_back_end"]'}
#     result = udf.transform(None, None, kvargs)
#     print(result)
