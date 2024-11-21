from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tritonclient.grpc as grpcclient
import numpy as np

# FastAPI app
app = FastAPI()

# Triton server configuration
TRITON_SERVER_URL = "localhost:1235"
MODEL_NAME = "doc2vec"

# Initialize the Triton client
try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Triton server: {e}")

# Input model for request validation
class InferenceRequest(BaseModel):
    texts: list[str]

@app.post("/infer")
async def infer_texts(request: InferenceRequest):
    try:
        # Extract batch of texts
        input_texts = request.texts

        # Prepare Triton input
        inputs = []
        inputs.append(grpcclient.InferInput("TEXT", [len(input_texts)], "BYTES"))
        text2np = np.array(input_texts, dtype=np.object_)
        inputs[0].set_data_from_numpy(text2np)

        # Prepare Triton output
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("VECTORIZE"))

        # Perform inference
        response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

        # Retrieve output
        vectorized_outputs = response.as_numpy("VECTORIZE")
        return {"vectors": vectorized_outputs.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
