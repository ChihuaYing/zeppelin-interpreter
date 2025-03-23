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

    def search_similarity(self, embedding, pattern):
        print("Searching similar embeddings")

        milvus_pattern = pattern.replace("*", "%")
        expr = 'path like "' + milvus_pattern + "'"
        entities = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            output_fields=["path"],
            limit=10,
            param={"metric_type": "COSINE"},
            expr=expr,
            search_params={
                "hints": "iterative_filter"
            }
        )

        return [hit.fields["path"] for hit in entities[0]]


class UDFSearchEmbedding:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFSearchEmbedding success：", kvargs)
        search_description = kvargs["description"].decode("utf-8")
        pattern = kvargs["pattern"].decode("utf-8")

        search_key_words = LLMDao().provide_keywords(search_description)
        print("key_words:", search_key_words)

        with Encoder() as ec:
            search_embedding = ec.encode([search_key_words])[0]

        similar_paths = MilvusDao(kvargs).search_similarity(search_embedding, pattern)

        return [["(path)"], ['BINARY']] + [
            [path.encode('utf-8')]
            for path in similar_paths
        ]