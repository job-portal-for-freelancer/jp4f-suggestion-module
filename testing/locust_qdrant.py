from locust import HttpUser, task, between
import random

class QdrantPerformanceTest(HttpUser):
    wait_time = between(1, 3)  # Wait time between requests

    @task
    def test_single_search(self):
        # Generate a random query vector for single search
        vector = [random.random() for _ in range(4)]  # Adjust dimension if needed
        payload = {"vector": vector}
        print("Payload single:", payload)
        self.client.post("/search", json=payload)

    @task
    def test_batch_search(self):
        # Generate random batch queries
        vectors = [
            {"vector": [random.random() for _ in range(4)]}
            for _ in range(random.randint(2, 5))
        ]

        payload = {"vectors": vectors}
        print("Payload batch:", payload)

        self.client.post("/search_batch", json=payload)


