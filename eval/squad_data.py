# Download, cache, parse, and sample the SQuAD 2.0 dev set.

import json
import random
from pathlib import Path

import requests
from tqdm import tqdm

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def download_squad(data_dir: Path = DEFAULT_DATA_DIR, url: str = SQUAD_URL, force: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "dev-v2.0.json"
    if dest.exists() and not force:
        return dest

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading SQuAD 2.0 dev set") as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def load_squad(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_contexts(raw: dict) -> list[dict]:
    """Flatten SQuAD's article -> paragraph -> qas structure into one entry per
    paragraph, each carrying its own real question set. plausible_answers are
    annotator-marked *incorrect* guesses for unanswerable questions and must
    never be treated as gold text, so they're dropped here."""
    contexts = []
    for article in raw["data"]:
        title = article["title"]
        for para_idx, paragraph in enumerate(article["paragraphs"]):
            questions = []
            for qa in paragraph["qas"]:
                seen = set()
                answers = []
                for a in qa.get("answers", []):
                    text = a["text"]
                    if text not in seen:
                        seen.add(text)
                        answers.append(text)
                questions.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "is_impossible": qa.get("is_impossible", False),
                    "answers": answers,
                })
            contexts.append({
                "title": title,
                "para_idx": para_idx,
                "context": paragraph["context"],
                "questions": questions,
            })

    contexts.sort(key=lambda c: (c["title"], c["para_idx"]))
    return contexts


def sample_contexts(contexts: list[dict], num_contexts: int, seed: int) -> list[dict]:
    if num_contexts > len(contexts):
        print(f"Warning: requested {num_contexts} contexts but only {len(contexts)} are available; using all of them.")
        num_contexts = len(contexts)
    return random.Random(seed).sample(contexts, num_contexts)
