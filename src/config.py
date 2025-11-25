# config.py

"""
Configuration file for language metadata.
This allows adding new languages without modifying the main script.

You can extend LANGUAGES by simply adding a new entry.
Each entry contains:
    - prefix: filename prefix used in test WAV files (e.g. "en_", "su_", ...)
    - label: the label index order used by the ML model

The model should load labels from this dict instead of hardcoding.
"""

LANGUAGES = [
    {
        "prefix": "su_",
        "name": "Finnish",
        "dataset": "xtreme_s/fleurs.fi_fi",
        "train_split": 7500,  # Increased from 5000 - more data for better learning
        "val_split": 1250,    # Increased from 1000 - better validation accuracy
        "weight": 1.0
    },
    {
        "prefix": "en_",
        "name": "English",
        "dataset": "xtreme_s/fleurs.en_us",
        "train_split": 7500,  # Increased from 5000
        "val_split": 1250,    # Increased from 1000
        "weight": 1.0
    },
    {
        "prefix": "es_",
        "name": "Spanish",
        "dataset": "xtreme_s/fleurs.es_419",
        "train_split": 7500,  # Increased from 5000
        "val_split": 1250,    # Increased from 1000
        "weight": 1.0
    },
    {
        "prefix": "ru_",
        "name": "Russian",
        "dataset": "xtreme_s/fleurs.ru_ru",
        "train_split": 7500,  # Increased from 5000
        "val_split": 1250,    # Increased from 1000
        "weight": 1.0
    },
]

# Derive model label list automatically from LANGUAGES
LABELS = [lang["name"] for lang in LANGUAGES]

# Build prefix → language map for true-language lookup
PREFIX_MAP = {lang["prefix"]: lang["name"] for lang in LANGUAGES}

LANGUAGE_WEIGHTS = {i: lang.get("weight", 1.0) for i, lang in enumerate(LANGUAGES)}

print(f"Optimized {len(LANGUAGES)} languages:")
for lang in LANGUAGES:
    print(f"   {lang['name']}: {lang['train_split']} train, {lang['val_split']} val, weight: {lang['weight']}")
