from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT_DIR / "archive/archive"
MODEL_ROOT = ROOT_DIR / "SmartPuzzleAI_merged_model_final"
SIAMESE_MODEL_PATH = MODEL_ROOT / "siamese_best.pth"
MERGED_SD_MODEL_PATH = MODEL_ROOT / "sd15_merged_kid_sketches"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = ROOT_DIR / "figures"

for directory in [OUTPUTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

EMOTION_KEYWORDS = [
    "angry",
    "happy",
    "sad",
    "fear",
    "fearful",
    "surprised",
    "neutral",
    "disgust",
]


def find_dataset_dir(dataset_root=DATASET_ROOT):
    actual_dataset_dir = None

    for root, dirs, _files in dataset_root.walk():
        emotion_folders = [d for d in dirs if d.lower() in EMOTION_KEYWORDS]
        if emotion_folders:
            actual_dataset_dir = root
            break

    if actual_dataset_dir:
        return actual_dataset_dir

    fallback = dataset_root / "NewArts2" / "NewArts2"
    if fallback.exists():
        return fallback

    return dataset_root


DATASET_DIR = find_dataset_dir()

