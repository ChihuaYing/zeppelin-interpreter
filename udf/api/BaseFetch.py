from abc import abstractmethod

import numpy as np

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
