import os

from DefaultLLM import LLMDao

class UDFDefaultGenerator:
    def __init__(self):
        pass

    def generate_by_udf_content(self,base_udf_content: str, description: str) -> str:
        prompt=f"""
            我是系统开发者，希望你为用户生成一个UDF，这个UDF的基类代码如下：
            ```python
            {base_udf_content}
            ```
            仅输出UDF子类的代码，不需要包含基类代码，请以纯文本格式输出代码，而不是 markdown 格式。
            请你根据以下用户提供描述，为用户生成一个UDF子类文件内容：
            ```
            {description}
            ```
            """
        print("Generating default UDF with prompt:\n", prompt)

        return LLMDao().request([prompt])[0]

    def generate(self,base_udf_name:str, description: str) -> str:
        baseDir = os.path.dirname(os.path.abspath(__file__))
        base_udf_file_name = 'Base'+ base_udf_name + '.py'
        base_udf_file_path = os.path.join(baseDir, base_udf_file_name)
        with open( base_udf_file_path, 'r') as f:
            base_udf = f.read()
        return self.generate_by_udf_content(base_udf,description)

if __name__ == "__main__":
    # get temp file path
    udf = UDFDefaultGenerator()
    result = udf.generate('Encoder','请使用Sentence Transformer的 all-MiniLM-L6-v2 进行编码')
    print(result)