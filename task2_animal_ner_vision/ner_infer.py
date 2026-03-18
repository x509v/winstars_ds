from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def load_model(model_source: str):
    """
    model_source can be:
    - local path: ./trained_models_ner
    - HF repo: username/model-name
    """
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForTokenClassification.from_pretrained(model_source)
    model.eval()
    return tokenizer, model


def extract_animal_entities(text: str, model_source: str, min_confidence: float = 0.5) -> List[str]:
    tokenizer, model = load_model(model_source)

    encoding = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    confidences, pred_ids = torch.max(probs, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    labels = [model.config.id2label[int(i)] for i in pred_ids[0]]
    confs = confidences[0].tolist()

    animal_tokens: List[str] = []
    current: List[str] = []
    for token, label, conf in zip(tokens, labels, confs):
        if token.startswith("##"):
            token = token[2:]
        if "ANIMAL" in label and conf >= min_confidence:
            if label.startswith("B-") and current:
                animal_tokens.append(" ".join(current))
                current = []
            current.append(token)
        else:
            if current:
                animal_tokens.append(" ".join(current))
                current = []
    if current:
        animal_tokens.append(" ".join(current))

    clean = []
    for ent in animal_tokens:
        ent = ent.replace("[CLS]", "").replace("[SEP]", "").strip()
        if ent:
            clean.append(ent)

    clean = list(dict.fromkeys(c.lower() for c in clean))
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NER inference to extract animal names from text.")
    parser.add_argument(
        "--model",
        type=str,
        help="Local path OR Hugging Face repo (e.g. username/model-name)",
        default="hesoyam3333/test_task_winstars"
    )
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    animals = extract_animal_entities(args.text, args.model, min_confidence=args.min_confidence)
    print(animals)


if __name__ == "__main__":
    main()