from __future__ import annotations

import argparse
from typing import List

from ner_infer import extract_animal_entities
from image_infer import predict_image_class


def check_statement(text: str, image_path: str, ner_model_dir: str, image_checkpoint: str) -> bool:
    """
    Full pipeline:
      1) Extract animal names from the text using NER.
      2) Classify the animal in the image.
      3) Return True if at least one animal from the text matches the image class.

    Matching is done via a simple case-insensitive substring check, e.g.
      - text: \"There is a cow in the picture\" -> NER -> [\"cow\"]
      - image prediction: \"cow\" -> True
    """
    text_animals: List[str] = extract_animal_entities(text, ner_model_dir)
    if not text_animals:
        return False

    image_label = predict_image_class(image_checkpoint, image_path)

    image_label_lower = image_label.lower()
    for animal in text_animals:
        if animal.lower() in image_label_lower or image_label_lower in animal.lower():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: text + image -> boolean (is the statement about the animal true?)."
    )
    parser.add_argument("--text", type=str, required=True, help="User text, e.g. 'There is a cow in the picture.'")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--ner-model-dir", type=str, required=True, help="Directory with fine-tuned NER model.")
    parser.add_argument("--image-checkpoint", type=str, required=True, help="Path to trained image classifier (.pt).")
    args = parser.parse_args()

    result = check_statement(
        text=args.text,
        image_path=args.image,
        ner_model_dir=args.ner_model_dir,
        image_checkpoint=args.image_checkpoint,
    )
    print(result)


if __name__ == "__main__":
    main()

