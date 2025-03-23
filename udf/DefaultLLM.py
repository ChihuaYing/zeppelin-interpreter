import asyncio

import aiohttp
from tenacity import retry, stop_after_attempt, wait_incrementing
import tqdm

class LLMDao:
    def __init__(self):
        self.url = "https://madmodel.cs.tsinghua.edu.cn/v1/chat/completions"
        self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0MjcxODA4MSwiZXhwIjoxNzQyNzM5NjgxfQ.Sos0C_cBaiEC_Ux4K0_sjwRbROFIGIBTyBP0gfDgW1A"
        self.model = "DeepSeek-R1-Distill-32B"

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
                "temperature": 0.0,
                # "max_tokens":100,
                "stream": False,
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

    def bulk_get_response(self, prompts: list[str], max_concurrent_requests: int = 20) -> list[str]:
        return asyncio.run(self._bulk_fetch(prompts, max_concurrent_requests))

    def bulk_get_response_without_think(self, prompts: list[str], max_concurrent_requests: int = 20) -> list[str]:
        responses = self.bulk_get_response(prompts, max_concurrent_requests)
        return [response.split("</think>")[-1].strip() for response in responses]

    def request(self, prompt: list[str]) -> [str]:
        return self.bulk_get_response_without_think(prompt)

if __name__ == "__main__":
    dao = LLMDao()
    prompts = [
        "What is the capital of France? don't output the thinking process",
        "Who is the president of the United States?",
        "What is the largest mammal in the world?",
    ]
    responses = dao.request(prompts)
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}\nResponse: {response}\n")
