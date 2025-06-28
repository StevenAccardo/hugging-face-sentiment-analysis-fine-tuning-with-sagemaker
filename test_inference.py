from sagemaker.huggingface import HuggingFacePredictor
from dotenv import load_dotenv
import os


load_dotenv()

endpoint_name = os.getenv('ENDPOINT_NAME')

if not endpoint_name:
  raise ValueError('ENDPOINT_NAME env var not set.')

predictor = HuggingFacePredictor(endpoint_name=endpoint_name)

input_text = {
  'inputs': "This movie was absolutely fantastic! Highly recommend it."
}

response = predictor.predict(input_text)

print(response)
[{'label': 'positive', 'score': 0.9968538880348206}]