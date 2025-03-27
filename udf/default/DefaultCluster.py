import asyncio
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_incrementing

import aiohttp
import tqdm
import numpy as np

from sklearn.cluster import KMeans

from api.BaseCluster import UDFBaseCluster, Node
from default.DefaultEncode import UDFDefaultEncode


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

class UDFDefaultCluster(UDFBaseCluster):

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

        ec = UDFDefaultEncode()
        current_embeddings = ec.encode(current_descriptions)
        return [
            Node(path, description, embedding, children)
            for path, description, embedding, children in
            zip(current_paths, current_descriptions, current_embeddings, children_list)
        ]

    def cluster(self, forest: list[Node], target: int) -> list[Node]:
        return self._nest_build_cluster(forest, target, 1)

if __name__ == "__main__":
    # Test the cluster algorithm
    from DefaultFetch import UDFDefaultFetch
    fetched = UDFDefaultFetch().transform([[],[]],[],{"pattern": "*".encode(), "level": "0".encode()})
    print(fetched)

    fetched[0] = [name[1:-1] for name in fetched[0]]

    clustered = UDFDefaultCluster().transform(fetched,[],{"target": "24".encode()})
    print(clustered)
