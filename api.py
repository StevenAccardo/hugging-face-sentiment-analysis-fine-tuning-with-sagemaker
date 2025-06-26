from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
sa_pipeline = pipeline('sentiment-analysis', model='./sentiment-model', tokenizer='./sentiment-model')

# Define the request body structure
class SAInput(BaseModel):
  sequence: list

@app.post('/model')
def ask(data: SAInput):
  result = sa_pipeline(inputs=data.sequence)
  return result