# Task 2 – Animal NER + Image Classification Pipeline

This folder contains the solution scaffold for **Task 2**: a two-stage ML pipeline that understands what animal the user mentions in text (NER) and checks whether it matches the animal in the provided image (image classification).

---

## High-level design

- **Dataset (image classification)**:
  - [Animal Image Dataset – 90 Different Animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) by Sourav Banerjee (Kaggle).
  - 5,400 images across **90 animal classes**, ~60 images per class.
  - See the [Dataset](#dataset) section below for full details.

- **NER model**:
  - Transformer-based token classification model (e.g. `distilbert-base-cased`) fine-tuned to detect animal mentions with labels `B-ANIMAL`, `I-ANIMAL`, `O`.

- **Image classification model**:
  - ResNet-18 (PyTorch) fine-tuned on the chosen animal dataset to predict a single animal class per image.

- **Pipeline**:
  1. NER extracts animal surface forms from the input text.
  2. Image classifier predicts the most probable animal class in the image.
  3. A simple string-matching heuristic compares extracted animals and predicted class and returns **True/False**.

---

## Files in this folder

| File | Description |
|---|---|
| `ner_train.py` | Parametrized training script for the NER model |
| `ner_infer.py` | Parametrized inference script for the NER model |
| `image_train.py` | Parametrized training script for the image classifier |
| `image_infer.py` | Parametrized inference script for the image classifier |
| `pipeline.py` | Wires NER + image classifier and returns a boolean |
| `requirements.txt` | Dependencies for this task |
| `eda_task2.ipynb` | Jupyter notebook for dataset EDA |

---

## Dataset

### Animal Image Dataset – 90 Different Animals

| Property | Value |
|---|---|
| **Source** | [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) |
| **Author** | Sourav Banerjee (`iamsouravbanerjee`) |
| **Total images** | 5,400 |
| **Classes** | 90 |
| **Images per class** | ~60 |
| **License** | See Kaggle dataset page |

#### Animal classes (90)

antelope, badger, bat, bear, bee, beetle, bison, boar, butterfly, cat, caterpillar, chimpanzee, cockroach, cow, coyote, crab, crow, deer, dog, dolphin, donkey, dragonfly, duck, eagle, elephant, flamingo, fly, fox, goat, goldfish, goose, gorilla, grasshopper, hamster, hare, hedgehog, hippopotamus, hornbill, horse, hummingbird, hyena, jellyfish, kangaroo, koala, ladybug, leopard, lion, lizard, lobster, mosquito, moth, mouse, octopus, okapi, orangutan, otter, owl, ox, oyster, panda, parrot, pelecan, penguin, pig, pigeon, porcupine, possum, raccoon, rat, reindeer, rhinoceros, rooster, salamander, seal, shark, sheep, snake, sparrow, squid, squirrel, starfish, swan, tiger, turkey, turtle, whale, wolf, wombat, woodpecker, zebra

> **Note:** The model published on Hugging Face (`hesoyam3333/test_task_winstars`) was trained on a **subset** of these 90 classes. Refer to the `class_to_idx` entry inside the checkpoint for the exact set used.

#### Downloading the dataset

```bash
# Option 1: Kaggle CLI
pip install kaggle
kaggle datasets download -d iamsouravbanerjee/animal-image-dataset-90-different-animals
unzip animal-image-dataset-90-different-animals.zip -d data/animals_raw

# Option 2: Manual download
# Visit https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
# and click the Download button (requires a free Kaggle account).
```

#### Preparing the directory layout

The dataset ships as a flat `Animals/` folder with one sub-folder per class. Split it into `train/` and `val/` before training:

```python
import os, shutil, random

SRC   = "data/animals_raw/Animals/Animals"
DEST  = "data/animals"
SPLIT = 0.8
random.seed(42)

for cls in os.listdir(SRC):
    images = os.listdir(os.path.join(SRC, cls))
    random.shuffle(images)
    cut = int(len(images) * SPLIT)
    for split, imgs in [("train", images[:cut]), ("val", images[cut:])]:
        out = os.path.join(DEST, split, cls)
        os.makedirs(out, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(SRC, cls, img), os.path.join(out, img))
```

After running this script the layout matches what `image_train.py` expects:

```
data/animals/
  train/
    antelope/  badger/  bat/  ...
  val/
    antelope/  badger/  bat/  ...
```

---

## Pre-trained models on Hugging Face

Both models are published on Hugging Face at **`hesoyam3333/test_task_winstars`** and are downloaded automatically if no local path is provided. You do **not** need to train the models yourself to run inference or the pipeline.

### NER model (`ner_infer.py`)

`ner_infer.py` accepts either a **local directory** or a **Hugging Face repo ID** via the `--model` argument.

```python
# Inside ner_infer.py — load_model()
tokenizer = AutoTokenizer.from_pretrained(model_source)
model = AutoModelForTokenClassification.from_pretrained(model_source)
```

`AutoTokenizer` and `AutoModelForTokenClassification` from 🤗 Transformers handle both cases transparently:
- If `model_source` is a local path that exists on disk, the files are loaded from there.
- If `model_source` is a Hugging Face repo ID (e.g. `hesoyam3333/test_task_winstars`), the model card, weights, and tokenizer files are downloaded automatically to the local Hugging Face cache (`~/.cache/huggingface/hub/`).

**Default repo used when `--model` is omitted:**
```
hesoyam3333/test_task_winstars
```

#### Inference — use the HF model (no local files needed)

```bash
python ner_infer.py \
  --text "There is a cow in the picture."
# --model defaults to hesoyam3333/test_task_winstars
```

#### Inference — use a local fine-tuned model

```bash
python ner_infer.py \
  --model models/ner_animals \
  --text "There is a cow in the picture."
```

Both commands print a Python list of extracted animal names, e.g. `["cow"]`.

---

### Image classification model (`image_infer.py`)

`image_infer.py` uses `hf_hub_download` from the `huggingface_hub` library to fetch the ResNet-18 checkpoint when no local path is provided.

```python
# Inside image_infer.py — resolve_checkpoint_path()
from huggingface_hub import hf_hub_download

DEFAULT_REPO = "hesoyam3333/test_task_winstars"
DEFAULT_MODEL_PATH_IN_REPO = "image_model/model.pt"

def resolve_checkpoint_path(checkpoint: str | None) -> str:
    if checkpoint and os.path.exists(checkpoint):
        return checkpoint                       # use local file
    print("⬇️ Downloading model from Hugging Face...")
    path = hf_hub_download(
        repo_id=DEFAULT_REPO,
        filename=DEFAULT_MODEL_PATH_IN_REPO,
    )
    return path
```

Resolution logic:
1. If `--checkpoint` points to a file that exists locally → use it directly.
2. Otherwise → download `image_model/model.pt` from `hesoyam3333/test_task_winstars` into the HF cache and use that path.

The downloaded `.pt` checkpoint contains:
- `model_state_dict` – ResNet-18 weights.
- `class_to_idx` – mapping from class name to integer index (reconstructed to `idx_to_class` at inference time).

#### Inference — auto-download from HF (no local checkpoint needed)

```bash
python image_infer.py \
  --image path/to/example.jpg
# --checkpoint is optional; omitting it triggers the HF download
```

#### Inference — use a local checkpoint

```bash
python image_infer.py \
  --checkpoint models/animals_resnet18.pt \
  --image path/to/example.jpg
```

Both commands print the predicted animal label, e.g. `cow`.

---

## NER: training and inference

### Input format for training

`ner_train.py` expects a **CSV** with at least:
- `text` – full sentence.
- `tags` – space-separated BIO tags aligned to `text.split()`.

Example row:

| text | tags |
|---|---|
| `There is a cow in the picture .` | `O O O B-ANIMAL O O O O` |

### Train command

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

### Inference command

```bash
python ner_infer.py \
  --model models/ner_animals \
  --text "There is a cow in the picture."
```

Output: `["cow"]`

---

## Image classification: training and inference

### Expected directory layout

```
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

Each subfolder name becomes a **class label**.

### Train command

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
- Fine-tunes a ResNet-18 (optionally using ImageNet pretraining).
- Saves a checkpoint containing `model_state_dict` and the `class_to_idx` mapping.

### Inference command

```bash
python image_infer.py \
  --checkpoint models/animals_resnet18.pt \
  --image path/to/example.jpg
```

Output: `cow`

---

## Full pipeline: text + image → boolean

`pipeline.py` combines both models:

```bash
python pipeline.py \
  --text "There is a cow in the picture." \
  --image path/to/example.jpg \
  --ner-model-dir models/ner_animals \
  --image-checkpoint models/animals_resnet18.pt
```

Internally:
1. `extract_animal_entities()` (from `ner_infer.py`) returns a list of animal names from the text.
2. `predict_image_class()` (from `image_infer.py`) returns the predicted animal label for the image.
3. Returns `True` if any extracted name matches the image label (case-insensitive substring match), otherwise `False`.

> Both `--ner-model-dir` and `--image-checkpoint` are optional. If omitted, models are downloaded automatically from `hesoyam3333/test_task_winstars`.

---

## EDA notebook (`eda_task2.ipynb`)

`eda_task2.ipynb` contains:
- Class distribution (counts per animal class).
- Sample images per class.
- Basic image statistics (image sizes, aspect ratios).
- Example texts and corresponding NER labels.

---
