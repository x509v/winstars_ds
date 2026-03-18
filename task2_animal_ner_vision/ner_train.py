from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, List, Dict

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


@dataclass
class NERTrainConfig:
    model_name: str
    train_file: str
    text_column: str
    tags_column: str
    output_dir: str
    num_train_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5


def load_data(cfg: NERTrainConfig) -> Dataset:
    """
    Load training data from a CSV file.

    Expected CSV columns:
      - text_column: raw sentence
      - tags_column: space-separated BIO tags aligned to tokens produced by str.split()

    Example:
      text: "There is a cow in the picture ."
      tags: "O O B-ANIMAL O O O O"
    """
    df = pd.read_csv(cfg.train_file)
    if cfg.text_column not in df.columns or cfg.tags_column not in df.columns:
        raise ValueError(
            f"Columns '{cfg.text_column}' and '{cfg.tags_column}' must be present in {cfg.train_file}."
        )
    return Dataset.from_pandas(df)


def tokenize_and_align_labels(
    examples: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    text_column: str,
    tags_column: str,
) -> Dict[str, Any]:
    all_tokens = [t.split() for t in examples[text_column]]
    all_tags = [t.split() for t in examples[tags_column]]

    tokenized = tokenizer(
        examples[text_column],
        truncation=True,
        is_split_into_words=False,
    )

    labels: List[List[int]] = []
    for i, word_tags in enumerate(all_tags):
        word_ids = tokenized.word_ids(batch_index=i)
        sample_labels: List[int] = []
        for word_id in word_ids:
            if word_id is None:
                sample_labels.append(-100)
            else:
                tag = word_tags[word_id]
                sample_labels.append(label2id[tag])
        labels.append(sample_labels)

    tokenized["labels"] = labels
    return tokenized


def train_ner(cfg: NERTrainConfig) -> None:
    dataset = load_data(cfg)

    # For simplicity we infer labels from the training file
    unique_tags = sorted(
        {tag for row in dataset[cfg.tags_column] for tag in str(row).split()}
    )
    label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2label = {idx: tag for tag, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "text_column": cfg.text_column,
            "tags_column": cfg.tags_column,
        },
    )

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


def parse_args() -> NERTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a transformer-based NER model for animal names."
    )
    parser.add_argument("--model-name", type=str, default="distilbert-base-cased")
    parser.add_argument("--train-file", type=str, required=True, help="Path to CSV with text and BIO tags.")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--tags-column", type=str, default="tags")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    return NERTrainConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        text_column=args.text_column,
        tags_column=args.tags_column,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    config = parse_args()
    train_ner(config)
    