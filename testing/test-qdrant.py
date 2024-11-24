from qdrant_client import QdrantClient, models

# Define Qdrant connection details
url_qdrant = r"https://a6ffce05-6187-4d07-908f-1f99523272b0.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key_qdrant = r"Njte5BIsSsLujzQKxUF3BfcUimvvH2V9GFEQ9FIcnk9NpCtqmwg1Lw"

# Connect to Qdrant
try:
    qdrant_client = QdrantClient(url=url_qdrant, api_key=api_key_qdrant)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Qdrant cluster: {e}")

# # Check the expected vector size
# collection_info = qdrant_client.get_collection("jp4f-vector")


# Resize the vectors to match the expected dimension
# Example: Pad with zeros to match a size of 100
# def pad_vector(vector, target_size):
#     return vector + [0] * (target_size - len(vector))

# vector_size = 100  # Replace with the actual vector size from collection_info
# points = [
#     models.PointStruct(
#         id=1,
#         payload={"color": "red"},
#         vector=pad_vector([0.9, 0.1, 0.1], 768),
#     ),
#     models.PointStruct(
#         id=2,
#         payload={"color": "green"},
#         vector=pad_vector([0.1, 0.9, 0.1], 768),
#     ),
#     models.PointStruct(
#         id=3,
#         payload={"color": "blue"},
#         vector=pad_vector([0.1, 0.1, 0.9], 768),
#     ),
# ]

# qdrant_client.upsert(collection_name="Sparse_Vector_DB", points=points)


qdrant_client.create_collection(
    collection_name="JP4F_Vector_DB",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        ),
    ),
)





# # # Update the collection configuration
qdrant_client.update_collection(
    collection_name="JP4F_Vector_DB",  # Replace with your actual collection name
    optimizers_config= models.OptimizersConfigDiff(indexing_threshold=10000),
    hnsw_config= models.HnswConfigDiff(
        m=64,
        ef_construct=512,
        full_scan_threshold=10000,
        max_indexing_threads=0,
        on_disk=False
    )
)





# vector_size = 100  # Replace with the actual vector size from collection_info
# points = [
#     models.PointStruct(
#         id=1,
#         payload={"projectId": 312},
#         vector=pad_vector([0.9, 0.1, 0.1], 768),
#     ),
#     models.PointStruct(
#         id=2,
#         payload={"projectId": 313},
#         vector=pad_vector([0.1, 0.9, 0.1], 768),
#     ),
#     models.PointStruct(
#         id=3,
#         payload={"projectId": 314},
#         vector=pad_vector([0.1, 0.1, 0.9], 768),
#     ),
# ]

# qdrant_client.upsert(collection_name="Sparse_Vector_DB", points=points)