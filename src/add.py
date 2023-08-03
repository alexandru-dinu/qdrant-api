import pickle
import random
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

collection_name = "glove"
glove_path = (
    Path.home() / "workspace/ml-data/code-gen/embeddings/glove.6B.200d.txt.pickle"
)

with open(glove_path, "rb") as fp:
    glove = pickle.load(fp)

words = list(glove.keys())
chosen_words = random.sample(words, 2)

client = QdrantClient("localhost", port=6333)
resp = client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=200, distance=Distance.COSINE),
)
print(resp)

points = [
    PointStruct(id=i, vector=glove[word], payload={"word": word})
    for i, word in enumerate(chosen_words)
]

client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points,
)
