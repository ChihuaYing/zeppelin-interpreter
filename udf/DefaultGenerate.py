import os

from DefaultLLM import LLMDao

class UDFDefaultGenerator:
    def __init__(self):
        pass

    def generate_by_udf_content(self,base_udf_content: str, description: str) -> str:
        prompt = f"""
            I am a system developer and I would like you to generate a UDF for the user. The base class code for the UDF is as follows:
            ```python
            {base_udf_content}
            ```
            Only output the UDF subclass code, without including the base class code. Please output the code in plain text format, not in markdown format.
            Based on the following description provided by the user, generate the content for the UDF subclass file:
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
    result = udf.generate('Encoder','Please encode using the Sentence Transformer model all-MiniLM-L6-v2')
    print(result)