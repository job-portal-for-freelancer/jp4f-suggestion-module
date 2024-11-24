from locust import HttpUser, task, between
import random
from faker import Faker

# Initialize Faker instance for generating random text
fake = Faker()

class TritonAPIUser(HttpUser):
    wait_time = between(1, 3)  # Time between requests

    @task
    def send_inference_request(self):
        # Randomize the number of texts in the batch (between 1 and 10)
        batch_size = random.randint(1, 10)
        
        # Generate random texts for the batch
        texts = [fake.sentence() for _ in range(batch_size)]
        
        # Prepare the payload
        payload = {
            "texts": texts
        }
        
        # Send a POST request to the /infer endpoint
        self.client.post("/infer-local", json=payload)
        self.client.post("/infer-triton", json=payload)

