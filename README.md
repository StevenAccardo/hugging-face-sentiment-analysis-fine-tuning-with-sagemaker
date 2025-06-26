# Hugging Face Sentiment Analysis Fine Tuning Example


## Steps to Use This Model and API
- ```pip install -r requirements.txt```
- ```python train.py``` This will train the model and place the trained model data in the qa-final-model/ directory. Wait for training to complete.
- In the terminal enter: ```uvicorn api:app --reload``` This will serve the API.
- Use the cURL below to make a request to the API.


### Results From Last Training

#### The first epoch is the selected checkpoint that the API will use as it had the lowest loss while maintaning comperable accuracy to the other epochs.
```                                                                      
{
  'eval_loss': 0.21767772734165192,
  'eval_accuracy': 0.9212,
  'eval_f1': 0.9216076402705929,
  'eval_runtime': 14.7899,
  'eval_samples_per_second': 169.034,
  'eval_steps_per_second': 21.163,
  'epoch': 1.0
}
{
  'eval_loss': 0.28838035464286804,
  'eval_accuracy': 0.924,
  'eval_f1': 0.9230145867098866,
  'eval_runtime': 12.8606,
  'eval_samples_per_second': 194.392,
  'eval_steps_per_second': 24.338,
  'epoch': 2.0
}
{
  'eval_loss': 0.35869795083999634,
  'eval_accuracy': 0.9276,
  'eval_f1': 0.9280889948351212,
  'eval_runtime': 12.9623,
  'eval_samples_per_second': 192.867,
  'eval_steps_per_second': 24.147,
  'epoch': 3.0
}
{
  'eval_loss': 0.4029352068901062,
  'eval_accuracy': 0.928,
  'eval_f1': 0.9271844660194175,
  'eval_runtime': 12.8943,
  'eval_samples_per_second': 193.884,
  'eval_steps_per_second': 24.274,
  'epoch': 4.0
} 
```

### Sample Postman Request

```
curl --location 'http://127.0.0.1:8000/model' \
--header 'Content-Type: application/json' \
--data '{
    "sequence": ["I loved that movie!", "That movie was not good."]
}'
```