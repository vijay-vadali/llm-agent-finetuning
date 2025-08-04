"""
Metaflow Training Pipeline - Automated fine-tuning workflow
Handles data preprocessing, model training, and evaluation.
"""

from metaflow import FlowSpec, step, Parameter
import pandas as pd
import logging
import traceback
from sklearn.metrics import accuracy_score, f1_score
import json
from datetime import datetime
#import pdb
import os
#from typing import Dict, Any
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForSequenceClassification,pipeline,TrainingArguments,Trainer,DataCollatorWithPadding
from datasets import Dataset
import torch

logging.basicConfig(level=logging.info)

class FineTuningFlow(FlowSpec):

    data_path = "/Users/vvadali/Documents/git/vadaliv/llm-agent-finetuning/examples/sample_data.csv"
    # data_path = Parameter('data_path', help='Path to training data CSV')
    model_name = Parameter(
        "model_name", help="Base model to fine-tune", default="distilbert-base-uncased"
    )
    task_type = Parameter(
        "task_type", help="Type of task", default="sequence_classification"
    )
    num_classes = Parameter("num_classes", help="Number of classes", default=2)
    num_epochs = Parameter("num_epochs", help="Training epochs", default=3)
    batch_size = Parameter("batch_size", help="Batch size", default=32)
    learning_rate = Parameter("learning_rate", help="Learning rate", default=2e-5)
    output_dir = Parameter(
        "output_dir", help="Output directory", default="./models/output/final_model"
    )

    @step
    def start(self):
        """Initialize the training pipeline."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            self.training_metrics = {}
            self.logs = []
            
        except Exception as e:
            logging.error("Metaflow initialization failed: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())
        self.next(self.load_data)
    @step
    def load_data(self):

        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"   Loaded {len(self.df)} samples")

            self.data_stats = {
                "num_samples": len(self.df),
                "num_classes": self.df["label"].nunique(),
                "avg_text_length": self.df["text"].str.len().mean(),
            }
        except Exception as e:
            logging.error("Data loading failed: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())
            raise
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        logging.info("🔧 Preprocessing data...")

        try:
            self.df["text"] = self.df["text"].astype(str)
            self.df["text"] = self.df["text"].str.strip()
            label_map = {"negative": 0, "positive": 1}
            self.df["label"] = self.df["label"].map(label_map)
            initial_count = len(self.df)
            self.df = self.df[self.df["text"].str.len() > 0]
            final_count = len(self.df)

            if initial_count != final_count:
                removed = initial_count - final_count
                logging.info(f"   Removed {removed} empty texts")

            self.train_df, self.val_df = train_test_split(
                self.df,
                test_size=0.2,
                random_state=42,
                stratify=(
                    self.df["label"] if len(self.df["label"].unique()) > 1 else None
                ),
            )

            logging.info(f"   Train samples: {len(self.train_df)}")
            logging.info(f"   Validation samples: {len(self.val_df)}")

            logging.info(
                f"Train/val split: {len(self.train_df)}/{len(self.val_df)}"
            )

        except Exception as e:
            logging.error("Preprocessing failed: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())
            raise

        self.next(self.train_model)

    @step
    def train_model(self):
        logging.info("⚙️ Training model...")

        try:

            logging.info(f"   Loading {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=int(self.num_classes)
            )

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"], truncation=True, padding=True, max_length=512
                )

            train_dataset = Dataset.from_pandas(self.train_df)
            val_dataset = Dataset.from_pandas(self.val_df)

            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)
            logging.info("train_dataset", train_dataset)
            logging.info("valid_dataset", val_dataset)
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=int(self.num_epochs),
                per_device_train_batch_size=int(self.batch_size),
                per_device_eval_batch_size=int(self.batch_size),
                learning_rate=float(self.learning_rate),
                eval_strategy="epoch",
                save_strategy="no",
                save_total_limit=1,
                load_best_model_at_end=False,
                logging_strategy="no"
            )

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = predictions.argmax(axis=-1)

                accuracy = accuracy_score(labels, predictions)
                f1 = f1_score(labels, predictions, average="weighted")

                return {"accuracy": accuracy, "f1": f1}

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            logging.info("   Starting training...")
            train_result = trainer.train()

            final_eval = trainer.evaluate(eval_dataset=val_dataset)

            self.final_f1_score = final_eval.get('eval_f1', 0.0)
            self.final_accuracy = final_eval.get('eval_accuracy', 0.0)

            model_path = self.output_dir
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)

            self.training_metrics = {
                "train_loss": train_result.training_loss,
                "epochs_completed": int(self.num_epochs),
                "model_path": model_path,
            }

            logging.info(f"   Training completed! Loss: {train_result.training_loss:.4f}")

        except Exception as e:
            logging.error("Training failed: %s", str(e))
            logging.debug("Traceback:\n%s", traceback.format_exc())
            raise

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """Evaluate the trained model."""
        logging.info("📈 Evaluating model...")

        try:

            model_path = self.training_metrics["model_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
            )

            val_texts = self.val_df["text"].tolist()
            val_labels = self.val_df["label"].tolist()
            label_to_id = {"LABEL_0": 0, "LABEL_1": 1}
            predictions = []
            for text in val_texts:
                result = classifier(text)[0]
                predicted_label = max(result, key=lambda x: x["score"])["label"]
                predictions.append(label_to_id[predicted_label])
            accuracy = accuracy_score(val_labels, predictions)
            self.evaluation_results = {
                "accuracy": accuracy,
                "validation_samples": len(val_labels),
                "model_path": model_path,
            }

            self.training_metrics["eval_accuracy"] = accuracy

            logging.info(f"   Evaluation accuracy: {accuracy:.4f}")

        except Exception as e:
            self.logs.append(f"Evaluation failed: {e}")
            logging.info(f"   Evaluation failed: {e}")

            # Set default values
            self.evaluation_results = {
                "accuracy": 0.0,
                "validation_samples": 0,
                "model_path": self.training_metrics.get("model_path", ""),
            }

        self.next(self.end)

    @step
    def end(self):
        logging.info("✅ Pipeline completed!")
        metrics = {
            "f1_score": float(self.final_f1_score),
            "accuracy": float(self.final_accuracy)
        }

        training_config = {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size
            #"dataset": self.dataset,
        }

        with open(f"{self.output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f)

        with open(f"{self.output_dir}/training_config.json", "w") as f:
            json.dump(training_config, f)
        self.final_result = {
            "status": "completed",
            "model_path": self.training_metrics.get("model_path", ""),
            "training_metrics": self.training_metrics,
            "evaluation_results": self.evaluation_results,
            "data_stats": self.data_stats,
            "logs": self.logs,
        }
        logging.info(f"Training complete! F1 Score: {self.final_f1_score:.3f}, Accuracy: {self.final_accuracy:.3f}")
        logging.info(f"   Model saved to: {self.training_metrics.get('model_path', 'N/A')}")
        logging.info(f"   Final accuracy: {self.evaluation_results.get('accuracy', 0):.4f}")