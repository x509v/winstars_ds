## Task 2 – Animal NER + Image Classification Pipeline

This folder contains the solution scaffold for **Task 2**: a two‑stage ML pipeline that
understands what animal the user mentions in text (NER) and checks whether it matches
the animal in the provided image (image classification).

### High‑level design

- **Dataset (image classification)**:
  - Use an animals dataset with **≥10 animal classes** arranged as an `ImageFolder`‑style
    directory, for example a dataset similar to *Animals-10*:
    - `dog`, `cat`, `horse`, `elephant`, `cow`, `sheep`, `butterfly`, `chicken`, `spider`, `squirrel`
  - Expected structure:
    - `data/animals/train/<class_name>/*.jpg`
    - `data/animals/val/<class_name>/*.jpg`

- **NER model**:
  - Transformer‑based token classification model (e.g. `distilbert-base-cased`) fine‑tuned
    to detect animal mentions with labels such as `B-ANIMAL`, `I-ANIMAL`, `O`.

- **Image classification model**:
  - ResNet‑18 (PyTorch) fine‑tuned on the chosen animal dataset to predict a single
    animal class per image.

- **Pipeline**:
  1. NER extracts animal surface forms from the input text.
  2. Image classifier predicts the most probable animal class in the image.
  3. A simple string‑matching heuristic compares extracted animals and predicted class
     and returns **True/False**.

### Files in this folder

- `ner_train.py` – parametrized training script for the NER model.
- `ner_infer.py` – parametrized inference script for the NER model.
- `image_train.py` – parametrized training script for the image classifier.
- `image_infer.py` – parametrized inference script for the image classifier.
- `pipeline.py` – Python script that wires NER + image classifier and returns a boolean.
- `requirements.txt` – dependencies for this task.
- `eda_task2.ipynb` – Jupyter notebook for dataset EDA.

### NER: training and inference

#### Input format for training

`ner_train.py` expects a **CSV** with at least:

- `text` – full sentence.
- `tags` – space‑separated BIO tags aligned to `text.split()`.

Example row:

```text
text,"There is a cow in the picture ."
tags,"O O B-ANIMAL O O O O"
```

You can create a small labeled dataset of animal‑related sentences using labels like
`B-ANIMAL` / `I-ANIMAL` for any animal name.

#### Train command

```bash
cd task2_animal_ner_vision
python ner_train.py \
  --train-file data/ner_train.csv \
  --text-column text \
  --tags-column tags \
  --model-name distilbert-base-cased \
  --output-dir models/ner_animals \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 5e-5
```

#### Inference command

```bash
python ner_infer.py \
  --model-dir models/ner_animals \
  --text "There is a cow in the picture."
```

This prints a Python list of extracted animal names, e.g. `["cow"]`.

### Image classification: training and inference

#### Expected directory layout

```text
data/animals/
  train/
    cow/
    horse/
    ...
  val/
    cow/
    horse/
    ...
```

Each subfolder name will become a **class label** (e.g. `cow`, `horse`, `elephant`, ...).

#### Train command

```bash
python image_train.py \
  --data-dir data/animals \
  --output-path models/animals_resnet18.pt \
  --batch-size 32 \
  --epochs 5 \
  --learning-rate 1e-3
```

This:

- Builds train/validation loaders from the directory.
- Fine‑tunes a ResNet‑18 (optionally using ImageNet pretraining).
- Saves a checkpoint containing:
  - `model_state_dict`
  - `class_to_idx` mapping.

#### Inference command

```bash
python image_infer.py \
  --checkpoint models/animals_resnet18.pt \
  --image path/to/example.jpg
```

This prints the predicted animal label, for example: `cow`.

### Full pipeline: text + image -> boolean

`pipeline.py` combines both models:

```bash
python pipeline.py \
  --text "There is a cow in the picture." \
  --image path/to/example.jpg \
  --ner-model-dir models/ner_animals \
  --image-checkpoint models/animals_resnet18.pt
```

Internally:

1. `extract_animal_entities` (from `ner_infer.py`) returns a list of animal names from text.
2. `predict_image_class` (from `image_infer.py`) returns the predicted animal label for the image.
3. The pipeline returns `True` if any extracted name matches the image label
   (case‑insensitive substring match), otherwise `False`.

### EDA notebook (`eda_task2.ipynb`)

You should create `eda_task2.ipynb` in this folder with, for example:

- Class distribution (counts per animal class).
- Sample images per class.
- Basic image statistics (e.g. image sizes, aspect ratios).
- A few example texts and corresponding NER labels (if you visualize the NER dataset).

### Notes

- The code is intentionally kept small and modular so you can easily swap datasets and
  model architectures.
- For a production‑grade system you would typically:
  - Use a larger and more diverse NER training set.
  - Add confidence thresholds and possibly top‑K predictions for the image classifier.
  - Implement more robust string matching (e.g. synonym lists, WordNet, or a mapping
    from class IDs to canonical animal names).

