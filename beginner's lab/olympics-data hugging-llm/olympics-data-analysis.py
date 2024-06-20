# olympic-data-analysis
# using zhipuai instead of openai

import pandas as pd
df = pd.read_csv("dataset/olympics_sections_text.csv")
# print(df.shape)
# print(df.head())

from zhipuai import ZhipuAI

zhipu_client = ZhipuAI(api_key="your api key")

def get_embedding_zhipu(inputs):
    if isinstance(inputs, str):
        inputs = [inputs]
    res = []
    for s in inputs:
        resp = zhipu_client.embeddings.create(
            input=s, model="embedding-2"
        )
        emd = resp.data[0]
        res.append(emd)
    return res

texts = [v.content for v in df.itertuples()]
print(len(texts))

# import pnlp

# emds = []
# for idx, batch in enumerate(pnlp.generate_batches_by_size(texts, 200)):
#     emd_data = get_embedding_zhipu(batch)
#     for v in emd_data:
#         emds.append(v.embedding)
#     print(f"batch: {idx} done")

# save emds to "emds1024.npz"
# import numpy as np
# np.savez("dataset/emds1024.npz", emds=emds)
# print("emds saved")


import numpy as np

arr = np.load("dataset/emds1024.npz")

emds = arr["emds"].tolist()

print(len(emds), len(emds[0]))

string = input()
print("continue...")

from qdrant_client import QdrantClient
qc_client = QdrantClient(host="localhost", port=6333)

from qdrant_client.models import Distance, VectorParams

qc_client.recreate_collection(
    collection_name="doc_qa",
    vectors_config=VectorParams(size=len(emds[0]), distance=Distance.COSINE),
)

payload=[
    {"content": v.content, "heading": v.heading, "title": v.title, "tokens": v.tokens} for v in df.itertuples()
]

qc_client.upload_collection(
    collection_name="doc_qa",
    vectors=emds,
    payload=payload
)

query = "Who won the 2020 Summer Olympics men's high jump?"
query_vector = get_embedding_zhipu(query)[0].embedding
hits = qc_client.search(
    collection_name="doc_qa",
    query_vector=query_vector,
    limit=5
)

print(hits)

string = input()
print("continue...")

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3

def construct_prompt(question: str):
    query_vector = get_embedding_zhipu(question)[0].embedding
    hits = qc_client.search(
        collection_name="doc_qa",
        query_vector=query_vector,
        limit=5
    )
    
    choose = []
    length = 0
    indexes = []
     
    for hit in hits:
        doc = hit.payload
        length += doc["tokens"] + separator_len
        if length > MAX_SECTION_LEN:
            break
            
        choose.append(SEPARATOR + doc["content"].replace("\n", " "))
        indexes.append(doc["title"] + doc["heading"])
            
    # Useful diagnostic information
    print(f"Selected {len(choose)} document sections:")
    print("\n".join(indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(choose) + "\n\n Q: " + question + "\n A:"

prompt = construct_prompt("Who won the 2020 Summer Olympics men's high jump?")

print("===\n", prompt)

def ask_zhipu(content):
    response = zhipu_client.chat.completions.create(
        model="glm-3-turbo", 
        messages=[{"role": "user", "content": content}],
        max_tokens=300,
        top_p=0.9,
    )
    ans = response.choices[0].message.content
    return ans

print(ask_zhipu(prompt))

query = "In the 2020 Summer Olympics, how many gold medals did the country which won the most medals win?"
prompt = construct_prompt(query)
answer = ask_zhipu(prompt)

print(f"\nQ: {query}\nA: {answer}")