from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def extract_animal_entities(text: str, model_dir: str, min_confidence: float = 0.5) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

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
            # B- means a NEW entity starts — close the previous span first
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

    # Deduplicate while preserving order, case-insensitive
    clean = list(dict.fromkeys(c.lower() for c in clean))
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NER inference to extract animal names from text.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to fine-tuned NER model directory.")
    parser.add_argument("--text", type=str, required=True, help="Input sentence.")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    animals = extract_animal_entities(args.text, args.model_dir, min_confidence=args.min_confidence)
    print(animals)


if __name__ == "__main__":
    main()

