from abc import abstractmethod

class UDFBaseDescribe:
    def __init__(self):
        pass

    @abstractmethod
    def describe(self, leaf_paths_with_type: dict[str, str]) -> dict[str, str]:
        pass

    def transform(self, data, args, kvargs):
        print(f"enter {self.__class__.__name__}")
        print(f"data[0]: {data[0]}")
        print(f"data[1]: {data[1]}")
        print(f"len(data): {len(data)}")
        print(f"args: {args}")
        print(f"kvargs: {kvargs}")

        pathIndex = data[0].index('path')
        typeIndex = data[0].index('type')
        leaf_paths_with_type = {row[pathIndex].decode('utf-8'): row[typeIndex].decode('utf-8') for row in data[2:]}
        paths_with_description = self.describe(leaf_paths_with_type)

        return [
            ['(path)','(type)','(description)'],
            ['BINARY','BINARY','BINARY'],
        ] + [
            [path.encode(), leaf_paths_with_type[path].encode() if path in leaf_paths_with_type else None, description.encode()]
            for path, description in paths_with_description.items()
        ]
