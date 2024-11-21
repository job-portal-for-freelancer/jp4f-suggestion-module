import tritonclient.http as httpclient
import numpy as np

# Define the server URL and model name
server_url = "localhost:1234"
model_name = "doc2vec"

# Create a Triton client
triton_client = httpclient.InferenceServerClient(url=server_url)

# Define the input text (example)
input_texts = "This is the first input text."

# Prepare the input data
inputs = []
inputs.append(httpclient.InferInput("TEXT", [1,], "BYTES"))

text2np = np.array([input_texts], dtype=np.object_)
inputs[0].set_data_from_numpy(text2np)

# Prepare the output data
outputs = []
outputs.append(httpclient.InferRequestedOutput("VECTORIZE"))

# Perform inference
results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

# Get the output data
vectorize = results.as_numpy("VECTORIZE")

# Print the resulting vector
print(f"Vectorized output: {vectorize}")
