from qdrant_client import QdrantClient, models


qdrant_client = QdrantClient(
    api_key="WbvUZU0wqIyJBwLHrbKI9mpiHRUCY1EAH--tdKdi2x0QX2tdoPoiTg",
    url="https://a6ffce05-6187-4d07-908f-1f99523272b0.us-east4-0.gcp.cloud.qdrant.io"
) 

qdrant_client.create_collection(
    collection_name="jp4f-vector-db",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE, on_disk=True),
    hnsw_config=models.HnswConfigDiff(on_disk=True),
    optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
)




qdrant_client.update_collection(
    collection_name="jp4f-vector-db",
    vectors_config={
        "my_vector": models.VectorParamsDiff(
            hnsw_config=models.HnswConfigDiff(
                m=32,
                ef_construct=123,
            ),
            quantization_config=models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=models.CompressionRatio.X32,
                    always_ram=True,
                ),
            ),
            on_disk=True,
        ),
    },
    hnsw_config=models.HnswConfigDiff(
        ef_construct=123,
    ),
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            always_ram=False,
        ),
    ),
)