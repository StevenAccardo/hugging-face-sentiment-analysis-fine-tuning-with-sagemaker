# Hugging Face Sentiment Analysis Fine Tuning with Sagemaker Example


## Steps to Use This Model and API
- ```pip install -r requirements.txt```
- Create a role in AWS that can be used with AWS Sagemaker
- Update the .env local file with the:
  - MODEL_URI - This is the image URI for the model you will use
  - AWS_ROLE - The ARN of the role you created earlier
  - ENDPOINT_NAME - The name of the endpoint you will create via Sagemaker
- ```python launch_training.py``` This will train the model and place the trained model data in an S3 bucket. Wait for training to complete.
- ```python deploy_model.py``` After the training is complete, create the model in Sagemaker and deploy an endpoint that leverages that model
- ```python test_inference.py``` This will ensure the model was created and the endpoint was deployed properly.
- In the terminal enter: ```uvicorn api:app --reload``` This will serve the API, which will allow you to send requests to the API locally and up to your Sagemaker endpoint behind the scenes.
- Use the cURL below to make a request to the API.

### Sample Postman Request

```
curl --location 'http://127.0.0.1:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "inputs": ["This is the best movie ever."]
}'
```