from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sagemaker.huggingface import HuggingFacePredictor
from dotenv import load_dotenv
import os


load_dotenv()

endpoint_name = os.getenv('ENDPOINT_NAME')

if not endpoint_name:
  raise ValueError('ENDPOINT_NAME env var not set.')

app = FastAPI()

class TextInput(BaseModel):
  inputs: list

endpoint_name = 'hf-sentiment-endpoint'  # must match what you deployed
predictor = HuggingFacePredictor(endpoint_name=endpoint_name)

@app.post('/predict')
async def predict_sentiment(data: TextInput):
  try:
    payload = {'inputs': data.inputs}
    response = predictor.predict(payload)
    return {'prediction': response}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# {
#   'prediction': [
#     {
#       'label': 'positive',
#       'score': 0.9954431056976318
#     }
#   ]
# }