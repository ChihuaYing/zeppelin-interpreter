import asyncio
import json
from collections import defaultdict
from typing import NamedTuple
from tenacity import retry, stop_after_attempt, wait_incrementing

import aiohttp
import shelve
import tqdm
import numpy as np

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
    def __init__(self, cache_path="embedding_cache.db"):
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


class Node(NamedTuple):
    path: str
    description: str
    embedding: np.ndarray
    children: list['Node'] = []


class Aggregator:
    def __init__(self, threshold=0.8, n_clusters=8, pca_components=50, tsne_components=3):
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.tsne_components = tsne_components

    def _build_cluster_labels(self, nodes: list[Node], cluster_target) -> np.array(np.int32):
        print("Building cluster for", len(nodes), "nodes")

        embeddings = [node.embedding for node in nodes]

        km = KMeans(n_clusters=cluster_target, random_state=0)
        km.fit(embeddings)
        return km.labels_

    def _nest_build_cluster(self, leaf_nodes: list[Node], target: int, depth: int) -> list[Node]:
        cluster_target = target ** depth

        if cluster_target >= len(leaf_nodes):
            print("Reached target depth", depth)
            return leaf_nodes

        print("Building nested cluster for", len(leaf_nodes), "nodes, target:", target)
        children_nodes = self._nest_build_cluster(leaf_nodes, target, depth + 1)

        labels = self._build_cluster_labels(children_nodes, cluster_target)

        clusters = defaultdict(list)
        for label, node in zip(labels, children_nodes):
            clusters[label].append(node)
        children_list = list(clusters.values())
        children_path_clusters = [[node.path for node in cluster] for cluster in children_list]
        children_description_clusters = [[node.description for node in cluster] for cluster in children_list]

        current_paths = LLMDao().summarize(children_description_clusters)
        current_descriptions = [
            ", ".join([current_path] + children_paths_cluster)
            for current_path, children_paths_cluster in zip(current_paths, children_path_clusters)
        ]

        ec = Encoder()  # 直接创建实例
        current_embeddings = ec.encode(current_descriptions)
        return [
            Node(path, description, embedding, children)
            for path, description, embedding, children in
            zip(current_paths, current_descriptions, current_embeddings, children_list)
        ]

    def build_cluster(self, leaf_nodes: list[Node], target: int) -> list[Node]:
        return self._nest_build_cluster(leaf_nodes, target, 1)


class UDFMerge:

    def _generate_result(self, nodes: list[Node], parent_path: list[str], results: list[list]):
        for node in nodes:
            if not node.children:
                results.append([node.path, ".".join(parent_path)])
            else:
                parent_path.append(node.path)
                self._generate_result(node.children, parent_path, results)
                parent_path.pop()

    def transform(self, data, args, kvargs):
        print("enter transform Merge success:", kvargs)
        pattern = kvargs["pattern"].decode("utf-8")

        paths_result = MilvusDao(kvargs).fetch_first_level(pattern)
        nodes = [Node(path, description, embedding) for path, embedding, description in paths_result]
        nested_clustered_nodes = Aggregator().build_cluster(nodes, 5)

        results = [["(path)", "(cluster)"], ['BINARY', 'BINARY']]
        self._generate_result(nested_clustered_nodes, [], results)
        print("results:", results)
        return results

# if __name__ == '__main__':
#     # 本地测试
#     udf = UDFMerge()
#     kvargs = {
#         'paths': b'["\xe6\xb5\xb7\xe6\xb4\x8b\xe5\xba\x95\xe8\xb4\xa8","\xe6\xb5\xb7\xe5\xba\x95\xe5\x9c\xb0\xe5\xbd\xa2","\xe5\xae\x9e\xe5\x86\xb5\xe5\x88\x86\xe6\x9e\x90\xe6\x95\xb0\xe6\x8d\xae","\xe5\x9b\xbd\xe5\xae\xb6\xe6\xb7\xb1\xe6\xb5\xb7\xe5\x9f\xba\xe5\x9c\xb0\xe7\xae\xa1\xe7\x90\x86\xe4\xb8\xad\xe5\xbf\x83","\xe4\xb8\x9c\xe6\xb5\xb7\xe4\xbf\xa1\xe6\x81\xaf\xe4\xb8\xad\xe5\xbf\x83","\xe9\x87\x8d\xe7\x82\xb9\xe7\xa0\x94\xe5\x8f\x91\xe8\xae\xa1\xe5\x88\x92","\xe5\x8c\x97\xe5\x86\xb0\xe6\xb4\x8b\xe5\x8d\xab\xe6\x98\x9f\xe9\x81\xa5\xe6\x84\x9f\xe4\xba\xa7\xe5\x93\x81\xe6\x95\xb0\xe6\x8d\xae","\xe6\xb5\xb7\xe6\xb4\x8b\xe6\xb0\x94\xe8\xb1\xa1","\xe6\xb5\xb7\xe6\xb4\x8b\xe5\x9c\xb0\xe7\x90\x83\xe7\x89\xa9\xe7\x90\x86","\xe9\xa3\x8e\xe7\x94\xb5\xe9\x81\xa5\xe6\x84\x9f\xe4\xba\xa7\xe5\x93\x81\xe6\x95\xb0\xe6\x8d\xae","\xe5\x8c\x97\xe6\xb5\xb7\xe4\xbf\xa1\xe6\x81\xaf\xe4\xb8\xad\xe5\xbf\x83","\xe5\x9b\xbd\xe5\xae\xb6\xe5\x8d\xab\xe6\x98\x9f\xe6\xb5\xb7\xe6\xb4\x8b\xe5\xba\x94\xe7\x94\xa8\xe4\xb8\xad\xe5\xbf\x83","\xe7\xbb\x9f\xe8\xae\xa1\xe5\x88\x86\xe6\x9e\x90\xe6\x95\xb0\xe6\x8d\xae","\xe6\xb5\xb7\xe6\xb4\x8b\xe6\xb0\xb4\xe6\x96\x87","\xe6\xb5\xb7\xe6\xb4\x8b\xe7\x94\x9f\xe7\x89\xa9","\xe5\x86\x8d\xe5\x88\x86\xe6\x9e\x90\xe6\x95\xb0\xe6\x8d\xae","\xe7\x9f\xa2\xe9\x87\x8f\xe5\x9c\xb0\xe5\x9b\xbe\xe6\x95\xb0\xe6\x8d\xae","\xe7\xa7\x91\xe6\x8a\x80\xe5\x9f\xba\xe7\xa1\x80\xe8\xb5\x84\xe6\xba\x90\xe8\xb0\x83\xe6\x9f\xa5\xe4\xb8\x93\xe9\xa1\xb9","\xe5\xbd\xb1\xe5\x83\x8f\xe9\x81\xa5\xe6\x84\x9f","\xe4\xb8\xad\xe5\x9b\xbd\xe8\xbf\x91\xe6\xb5\xb7\xe7\x8e\xaf\xe5\xa2\x83\xe9\x81\xa5\xe6\x84\x9f\xe4\xba\xa7\xe5\x93\x81\xe6\x95\xb0\xe6\x8d\xae"]'
#     }
#     result = udf.transform(None, None, kvargs)
#     print(result)
