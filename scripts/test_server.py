import requests

def test_streaming():
    url = "http://localhost:8000/generate"
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    <antThinking>
    You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
    </antThinking>

    Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    <thinking>
    """
    
    with requests.post(url, json={"prompt": prompt}, stream=True) as response:
        if response.status_code == 200:
            print("Streaming response:")
            for chunk in response.iter_content(chunk_size=1):
                if chunk:
                    print(chunk.decode('utf-8'), end='', flush=True)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    test_streaming()