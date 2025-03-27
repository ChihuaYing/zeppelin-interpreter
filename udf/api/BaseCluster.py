from abc import abstractmethod
from typing import NamedTuple

import numpy as np

class Node(NamedTuple):
    path: str
    description: str
    embedding: np.ndarray
    children: list['Node'] = []

class UDFBaseCluster:
    @abstractmethod
    def cluster(self, forest: list[Node], target: int) -> list[Node]:
        pass

    def _generate_result(self, nodes: list[Node], parent_path: list[str], results: list[list]):
        for node in nodes:
            if not node.children:
                results.append([node.path, ".".join(parent_path)])
            else:
                parent_path.append(node.path)
                self._generate_result(node.children, parent_path, results)
                parent_path.pop()

    def transform(self, data, args, kvargs):
        print(f"enter {self.__class__.__name__}")
        print(f"data[0]: {data[0]}")
        print(f"data[1]: {data[1]}")
        print(f"len(data): {len(data)}")
        print(f"args: {args}")
        print(f"kvargs: {kvargs}")

        target= int(kvargs["target"].decode("utf-8"))

        path_index = data[0].index('path')
        description_index = data[0].index('description')
        embedding_index = data[0].index('embedding')

        paths = [row[path_index].decode('utf-8') for row in data[2:]]
        descriptions = [row[description_index].decode('utf-8') for row in data[2:]]
        embeddings = [np.frombuffer(row[embedding_index], dtype=np.float32) for row in data[2:]]

        nodes = [Node(path, description, embedding) for path, description, embedding in zip(paths, descriptions, embeddings)]

        nested_clustered_nodes = self.cluster(nodes,target)

        results = [["(path)", "(cluster)"], ['BINARY', 'BINARY']]
        self._generate_result(nested_clustered_nodes, [], results)
        print("results:", results)
        return results
