import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

client = QdrantClient(host="localhost", port=6333)

from gensim.models import Word2Vec

example_text = [["Dies", "ist", "ein", "Beispieltext."], ["Ein", "weiterer", "Text."]]

model = Word2Vec(example_text, min_count=1)
vectors = model.wv.vectors


index_name = "my_index"
collection_name = "my_collection"
payload = {"text": "Beispieltext"}


points = []
for i, text in enumerate(vectors):
    point = PointStruct(
        id = i + 1,
        vector = vectors[i],
        payload = {"text": text}
    )
    points.append(point)

client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=100, distance=Distance.COSINE),
)
print(points)

client.upsert(
    collection_name="my_collection",
    points=points
)


collection_info = client.get_collection(collection_name="my_collection")

query_vector = np.random.rand(100)
hits = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    limit=5  # Return 5 closest points
)

print(hits)