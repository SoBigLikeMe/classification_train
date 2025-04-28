from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import BertForSequenceClassification

# Load the MRPC dataset
raw_datastest = load_dataset("csv", data_files={"train": "/data/MRPC/train.tsv", "dev": "/data/MRPC/dev.tsv"})

# Load the BERT tokenizer and model
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the BERT model for sequence classification
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# Tokenize the dataset
tokenized_datasets = raw_datastest.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Remove unnecessary columns and rename the label column
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

# Define the data collator
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
# Load the model

model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)