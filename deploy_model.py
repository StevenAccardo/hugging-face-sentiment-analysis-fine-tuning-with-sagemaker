from sagemaker.session import Session
from sagemaker.huggingface import HuggingFaceModel
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize SageMaker session and role
session = Session()
role = os.getenv('AWS_ROLE')

# Location of trained model artifacts from training session
model_uri = os.getenv('MODEL_URI')

if not role and model_uri :
  raise ValueError('Missing env vars.')

# Define the Hugging Face model
huggingface_model = HuggingFaceModel(
  model_data=model_uri,
  role=role,
  transformers_version='4.26.0', # Must match supported inference version
  pytorch_version='1.13.1', # Must match supported inference version
  py_version='py39', # Must match supported inference version
  # entry_point='inference.py', # Custom inference logic. Only needed if we need pre or post processing for input sequences to the model. In this case, we do not.
)

# Deploy model to an endpoint
predictor = huggingface_model.deploy(
  initial_instance_count=1,
  instance_type='ml.m5.large',
  endpoint_name='hf-sentiment-endpoint'
)

print('Endpoint deployed at:', predictor.endpoint_name)
