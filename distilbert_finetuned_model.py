import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import os

class Model:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.model_path = "./finetuned_model"
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, X, y=None):
        data_dict = {"text": X}
        if y is not None:
            data_dict["labels"] = self.label_encoder.transform(y)

        dataset = Dataset.from_dict(data_dict)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)
        dataset = dataset.remove_columns(["text"])
        dataset.set_format("torch")
        return dataset

    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    def fit(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        if os.path.exists(self.model_path):
            print("📦 Loading saved model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        else:
            dataset = self.preprocess_data(X, y)
            dataset = dataset.train_test_split(test_size=0.1)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(self.label_encoder.classes_),
                id2label={i: label for i, label in enumerate(self.label_encoder.classes_)},
                label2id={label: i for i, label in enumerate(self.label_encoder.classes_)}
            )

            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_dir="./logs",
                fp16=torch.cuda.is_available(),
                logging_steps=10,
                report_to="none"  # 👈 disable wandb
            )


            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                compute_metrics=self.compute_metrics
            )

            self.trainer.train()

            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)

        self.trainer = Trainer(model=self.model, tokenizer=self.tokenizer)

    def predict(self, X):
        dataset = self.preprocess_data(X)
        predictions = self.trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        return self.label_encoder.inverse_transform(preds)
