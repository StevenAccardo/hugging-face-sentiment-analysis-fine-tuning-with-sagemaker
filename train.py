from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. Load IMDB dataset (already split into 'train' and 'test')
# Dataset doesn't come with validation/eval dataset, so needed to split out training dataset to create one. Or could have used test dataset, but rather save that for actual testing
dataset = load_dataset('imdb')
split_dataset = dataset['train'].train_test_split(test_size=0.1)
dataset['train'] = split_dataset['train']
dataset['validation'] = split_dataset['test']

# print(dataset['train'][0])
{
  'text': "Being a huge fan of the Japanese singers Gackt and Hyde, I was terribly excited when I found out that they had made a film together and made it my mission in life to see it. I was not disappointed. In fact, this film greatly exceeded my expectations. Knowing that both Gackt and Hyde are singers rather than actors, I was prepared for brave yet not really that fulfilling performances, but am delighted to say that both of them managed to keep me captivated and believing the story as it went on. Moon Child has just the right amount of humour, action, romance and serious, heart-wrenching moments. I can't say that I've ever cried more at a film and these more tender moments are admirably acted by the pair, in my opinion, definitely proving their skills as actors. The fight scenes are absolutely stunning and although there are a few moments of uncertainty to begin with, you are quick to get into the movie and begin to bond with the characters. I thoroughly recommend this film to anyone, especially those who are fans of Gackt and Hyde.",
  'label': 1
}

# 2. Load tokenizer and model (binary classification: 2 labels)
checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# num_labels arg adjusts the classification head to have the right number of output labels — in this case, 2 for binary sentiment classification (positive/negative).
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Manually add the classification labels since the sentiment analysis labels don't come with the base model
label2id = {'negative': 0, 'positive': 1}
id2label = {0: 'negative', 1: 'positive'}

model.config.label2id = label2id
model.config.id2label = id2label
tokenizer.label2id = label2id
tokenizer.id2label = id2label

# 3. Preprocessing function (tokenizes each text input)
def preprocess_function(examples):
  return tokenizer(examples['text'], padding=False, truncation=True)

# 4. Apply preprocessing to dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 5. Set format for PyTorch and applyt to the listed columns
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# print(tokenized_dataset)
{
  'train': {
    'features': ['text', 'label', 'input_ids', 'attention_mask'],
    'num_rows': 22500
  },
  'test': {
      'features': ['text', 'label', 'input_ids', 'attention_mask'],
      'num_rows': 25000
  },
  'unsupervised': {
      'features': ['text', 'label', 'input_ids', 'attention_mask'],
      'num_rows': 50000
  },
  'validation': {
      'features': ['text', 'label', 'input_ids', 'attention_mask'],
      'num_rows': 2500
  }
}

# 6. Compute metrics function for evaluation
def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  acc = accuracy_score(labels, predictions)
  f1 = f1_score(labels, predictions)
  return {
    'accuracy': acc,
    'f1': f1
  }

# 7. Pass dataset patches through the datacollator
from transformers import DataCollatorWithPadding
# Dynamically pad sequences to the longest sequence length in the batch
# Prevents padding all sequences to the dataset’s longest example
# Saves memory and speeds up training
# Is best practice for variable-length text input
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 8. Training arguments
# I've chosen 4 epochs since we have the load_best_model_at_end option checked
training_args = TrainingArguments(
  output_dir='./training-output', # The output directory where the model predictions and checkpoints will be written.
  eval_strategy='epoch', # When to run evaluation: every 'epoch', 'steps', or 'no'
  save_strategy='epoch', # How often the model weights and etc are saved
  logging_dir='./training-logs', # Where to write logs (for TensorBoard or debugging)
  learning_rate=2e-5, # Base learning rate for the optimizer (AdamW by default)
  per_device_train_batch_size=8, # Batch size per GPU (or CPU) during training
  per_device_eval_batch_size=8, # Batch size per GPU (or CPU) during evaluation
  num_train_epochs=4, # Number of full passes through the training dataset (Default is 3)
  weight_decay=0.01, # L2 regularization: helps prevent overfitting
  load_best_model_at_end=True, # At end of training, load the checkpoint with best eval score
  metric_for_best_model='eval_loss', # Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`.
  greater_is_better=False, # Related to the metric_for_best_model, tells the trainer that the best values are lower, not higher.
  max_grad_norm=1.0  # <- this clips gradient norm to 1.0
)

# 9. Trainer instance
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset['train'],
  eval_dataset=tokenized_dataset['validation'],
  tokenizer=tokenizer,
  data_collator=data_collator, # Determines what happens to the data batches after tokenization, but right before being passed to the trainer.
  compute_metrics=compute_metrics
)

# 10. Train the model
trainer.train()

# 11. Evaluate the final model on test set
# Uncomment if another evaluation is needed
# eval_results = trainer.evaluate()
# print('Evaluation results:', eval_results)

# 12. Test the resulting model using the test dataset
# test_results = trainer.evaluate(tokenized_dataset['test'])
# print('Tests results:', test_results)

# 13. Save model and tokenizer (optional)
model.save_pretrained('./sentiment-model')
tokenizer.save_pretrained('./sentiment-model')
