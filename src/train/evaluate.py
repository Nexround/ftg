from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("./lora_bert_imdb")
model = BertForSequenceClassification.from_pretrained("./lora_bert_imdb", num_labels=2, cache_dir="/cache/huggingface/hub")

# Apply LoRA configuration again
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["query", "key"], lora_dropout=0.1, bias="none"
)
model = get_peft_model(model, lora_config)

# Load the IMDB test dataset
dataset = load_dataset("imdb", cache_dir="/cache/huggingface/datasets")

# Tokenize the test dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )

tokenized_test = dataset["test"].map(tokenize_function, batched=True)
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define the compute_metrics function
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Setup TrainingArguments for evaluation
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save evaluation results
    per_device_eval_batch_size=16,  # Evaluation batch size
    logging_dir="./logs",  # Directory for logs
    label_names=["label"],
)

# Setup Trainer with the model and evaluation data
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    
    compute_metrics=lambda p: {
        'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean()
    },
)

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
