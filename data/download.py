"""
data/download.py — Dataset Downloader  [V1 - HF WN18RR Update]

Downloads all supported KGE benchmark datasets and builds vocabulary mappings.
WN18RR has been updated to download safely via Hugging Face to bypass GitHub
case-sensitivity and 400 errors.

SUPPORTED DATASETS:
    fb15k237  FB15k-237 (Toutanova & Chen 2015). 14,541 entities, 237 relations.
              Standard benchmark. Primary comparison target.
    wn18rr    WN18RR (Dettmers et al. 2018). 40,943 entities, 11 relations.
              Requires ChunkedEvaluator during evaluation. (Downloads via HF)
    nell995   NELL-995 (Xiong et al. 2017). 75,492 entities, 200 relations.
              Specifically designed for multi-hop path reasoning.
              Required for A* conferences if your model is path-based.
    yago3_10  YAGO3-10 (Mahdisoltani et al. 2013). 123,182 entities, 37 relations.
              Scalability test. Path cache build: 2-6 hours.
    codex_m   CoDEx-M (Safavi & Koutra 2020). 17,050 entities, 51 relations.
              Cleaner, harder than FB15k-237. Preferred by newer reviewers.
    codex_l   CoDEx-L. 77,951 entities, 69 relations.

USAGE:
    python data/download.py --dataset fb15k237 --output_dir data/raw
    python data/download.py --dataset wn18rr    # Now uses Hugging Face
    python data/download.py --dataset yago3_10  # takes longest
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

# ── Dataset registry ─────────────────────────────────────────────────────────

DATASET_URLS: dict[str, dict] = {
    "fb15k237": {
        "url":        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/",
        "files":      ["train.txt", "valid.txt", "test.txt"],
        "dir_name":   "fb15k237",
        "description": "14,541 entities, 237 relations, ~272k train triples",
    },
    "wn18rr": {
        "url":        "HF: VLyb/WN18RR", 
        "files":      ["train.txt", "valid.txt", "test.txt"],
        "dir_name":   "wn18rr",
        "description": "40,943 entities, 11 relations, ~87k train triples",
    },
    "nell995": {
        "url":        "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/NELL-995/",
        "files":      ["train.txt", "valid.txt", "test.txt"],
        "dir_name":   "nell995",
        "description": "75,492 entities, 200 relations, ~150k train triples",
    },
    "yago3_10": {
        "url":        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/",
        "files":      ["train.txt", "valid.txt", "test.txt"],
        "dir_name":   "yago3_10",
        "description": "123,182 entities, 37 relations, ~1.08M train triples",
    },
    "codex_m": {
        "url":        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/",
        "files":      ["train.txt", "valid.txt", "test.txt"],
        "dir_name":   "codex_m",
        "description": "17,050 entities, 51 relations, ~185k train triples",
    },
    "codex_l": {
        "url":        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l/",
        "files":      ["train.txt", "valid.txt", "test.txt"],
        "dir_name":   "codex_l",
        "description": "77,951 entities, 69 relations, ~612k train triples",
    },
}

# ── Download helpers ──────────────────────────────────────────────────────────

def download_file(url: str, dest_path: Path, verbose: bool = True) -> bool:
    """
    Download a single file. Returns True on success.

    Uses requests if available, falls back to urllib.
    Shows progress for large files (>1MB).
    """
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        if verbose:
            print(f"  Already exists: {dest_path.name}")
        return True

    try:
        # Try requests first (better progress display)
        try:
            import requests
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if verbose and total > 0 and downloaded % (1024 * 1024) < 8192:
                        pct = 100 * downloaded / total
                        print(f"\r  Downloading {dest_path.name}: {pct:.0f}%", end="", flush=True)
            if verbose and total > 1024 * 1024:
                print()
        except ImportError:
            # Fallback: urllib
            if verbose:
                print(f"  Downloading {dest_path.name}...", end="", flush=True)
            urllib.request.urlretrieve(url, dest_path)
            if verbose:
                print(" done")

        return True

    except Exception as e:
        if verbose:
            print(f"\n  ERROR downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_dataset(
    dataset_name: str,
    output_dir:   str | Path = "data/raw",
    verbose:      bool       = True,
) -> Path:
    """
    Download a complete dataset (train.txt, valid.txt, test.txt).

    Args:
        dataset_name: One of the keys in DATASET_URLS.
        output_dir:   Local directory to save files.
        verbose:      Print download progress.

    Returns:
        Path to the dataset directory containing train/valid/test files.

    Raises:
        ValueError: If dataset_name is not supported.
        RuntimeError: If download fails.
    """
    if dataset_name not in DATASET_URLS:
        valid = list(DATASET_URLS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Choose from: {valid}"
        )

    config   = DATASET_URLS[dataset_name]
    dir_name = config["dir_name"]
    dest_dir = Path(output_dir) / dir_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ─── HUGGING FACE INTERCEPT (Dynamic) ─────────────────────────────────────
    if config["url"].startswith("HF:"):
        hf_repo = config["url"].split("HF:")[1].strip()
        if verbose:
            print(f"\nDownloading {dataset_name} from Hugging Face ({hf_repo}): {config['description']}")
            print(f"  Destination: {dest_dir}")
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to download this dataset. "
                "Please install it by running: pip install datasets"
            )
            
        hf_dataset = load_dataset(hf_repo)
        split_map = {
            "train": "train.txt",
            "validation": "valid.txt", 
            "test": "test.txt"
        }
        
        for hf_split, filename in split_map.items():
            if hf_split in hf_dataset:
                filepath = dest_dir / filename
                if not filepath.exists():
                    if verbose:
                        print(f"  Saving {hf_split} split to {filename}...")
                    with open(filepath, "w", encoding="utf-8") as f:
                        for row in hf_dataset[hf_split]:
                            f.write(f"{row['head']}\t{row['relation']}\t{row['tail']}\n")
                else:
                    if verbose:
                        print(f"  Already exists: {filename}")
                        
        if verbose:
            print(f"  ✓ {dataset_name} downloaded and formatted in {dest_dir}")
        return dest_dir
    # ──────────────────────────────────────────────────────────────────────────

    # Standard URL download for all other datasets
    base_url = config["url"]
    if verbose:
        print(f"\nDownloading {dataset_name}: {config['description']}")
        print(f"  Destination: {dest_dir}")

    success = True
    for fname in config["files"]:
        url      = base_url + fname
        dest     = dest_dir / fname
        ok       = download_file(url, dest, verbose=verbose)
        success  = success and ok

    if not success:
        raise RuntimeError(
            f"Failed to download some files for {dataset_name}. "
            f"Check your internet connection or download manually from:\n  {base_url}"
        )

    if verbose:
        print(f"  ✓ {dataset_name} downloaded to {dest_dir}")

    return dest_dir


# ── Vocabulary building ───────────────────────────────────────────────────────

def load_entity_relation_maps(
    dataset_dir: str | Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Build entity2id and relation2id from all split files.

    Reads train.txt, valid.txt, test.txt and collects all unique
    entities and relations. Assigns integer IDs in order of first appearance.

    This is the standard KGE preprocessing step. IDs are consistent
    across all splits (same entity always gets same ID).

    Args:
        dataset_dir: Path containing train.txt, valid.txt, test.txt.

    Returns:
        (entity2id, relation2id) dicts mapping string → int.

    Example:
        >>> entity2id, relation2id = load_entity_relation_maps("data/raw/fb15k237")
        >>> print(f"Entities: {len(entity2id)}, Relations: {len(relation2id)}")
        Entities: 14541, Relations: 237
    """
    dataset_dir = Path(dataset_dir)
    entities: dict[str, int] = {}
    relations: dict[str, int] = {}

    for split_file in ["train.txt", "valid.txt", "test.txt"]:
        fpath = dataset_dir / split_file
        if not fpath.exists():
            continue

        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                head, rel, tail = parts

                if head not in entities:
                    entities[head] = len(entities)
                if tail not in entities:
                    entities[tail] = len(entities)
                if rel not in relations:
                    relations[rel] = len(relations)

    return entities, relations


def build_adjacency_from_file(
    triple_file:  str | Path,
    entity2id:    dict[str, int],
    relation2id:  dict[str, int],
) -> dict[int, list[tuple[int, int]]]:
    """
    Build KG adjacency dict from a triple file.

    For use with PathCacheBuilder before training.

    Args:
        triple_file:  Path to train.txt.
        entity2id:    Entity string → ID mapping.
        relation2id:  Relation string → ID mapping.

    Returns:
        {entity_id: [(relation_id, neighbor_entity_id), ...]}
    """
    adj: dict[int, list] = {i: [] for i in range(len(entity2id))}

    with open(triple_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if h in entity2id and r in relation2id and t in entity2id:
                h_id = entity2id[h]
                r_id = relation2id[r]
                t_id = entity2id[t]
                adj[h_id].append((r_id, t_id))

    return adj


def save_vocab(
    entity2id:   dict[str, int],
    relation2id: dict[str, int],
    output_dir:  str | Path,
) -> None:
    """Save entity2id and relation2id to tsv files for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "entity2id.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(entity2id)}\n")
        for entity, idx in sorted(entity2id.items(), key=lambda x: x[1]):
            f.write(f"{entity}\t{idx}\n")

    with open(output_dir / "relation2id.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(relation2id)}\n")
        for rel, idx in sorted(relation2id.items(), key=lambda x: x[1]):
            f.write(f"{rel}\t{idx}\n")

    print(f"Saved vocab: {len(entity2id)} entities, {len(relation2id)} relations → {output_dir}")


def load_vocab(
    vocab_dir: str | Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """Load pre-built entity2id and relation2id from tsv files."""
    vocab_dir = Path(vocab_dir)
    entity2id: dict[str, int] = {}
    relation2id: dict[str, int] = {}

    for target, path, mapping in [
        ("entity2id", vocab_dir / "entity2id.txt", entity2id),
        ("relation2id", vocab_dir / "relation2id.txt", relation2id),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run download.py first.")
        with open(path, encoding="utf-8") as f:
            count = int(f.readline().strip())
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    mapping[parts[0]] = int(parts[1])

    return entity2id, relation2id


# ── Preprocessing ─────────────────────────────────────────────────────────────

def get_dataset_stats(dataset_dir: str | Path) -> dict:
    """
    Compute and return dataset statistics.

    Returns dict with num_entities, num_relations, num_triples per split.
    """
    dataset_dir = Path(dataset_dir)
    entity2id, relation2id = load_entity_relation_maps(dataset_dir)

    stats: dict = {
        "num_entities":  len(entity2id),
        "num_relations": len(relation2id),
    }

    for split in ["train", "valid", "test"]:
        fpath = dataset_dir / f"{split}.txt"
        if fpath.exists():
            with open(fpath) as f:
                n = sum(1 for line in f if line.strip() and len(line.split("\t")) == 3)
            stats[f"num_{split}_triples"] = n

    # Memory and time estimates
    E = len(entity2id)
    stats["path_cache_estimate_min"] = (
        0.1 if E < 100 else        # toy
        30  if E < 20000 else      # fb15k237
        60  if E < 50000 else      # wn18rr, nell995
        240                        # yago3_10
    )
    stats["chunked_evaluator_required"] = E > 20000

    return stats


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download KGE benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k}: {v['description']}"
            for k, v in DATASET_URLS.items()
        ),
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_URLS.keys()),
        help="Dataset name to download.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/raw",
        help="Output directory (default: data/raw).",
    )
    parser.add_argument(
        "--save_vocab", action="store_true",
        help="Also save entity2id.txt and relation2id.txt.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print dataset statistics after download.",
    )
    args = parser.parse_args()

    # Download
    dataset_dir = download_dataset(args.dataset, args.output_dir)

    # Build and optionally save vocabulary
    entity2id, relation2id = load_entity_relation_maps(dataset_dir)

    if args.save_vocab:
        save_vocab(entity2id, relation2id, dataset_dir)

    if args.stats:
        stats = get_dataset_stats(dataset_dir)
        print(f"\nDataset statistics for {args.dataset}:")
        for k, v in stats.items():
            if isinstance(v, bool):
                print(f"  {k}: {'YES (required)' if v else 'no'}")
            elif isinstance(v, float):
                print(f"  {k}: ~{v:.0f} minutes")
            else:
                print(f"  {k}: {v:,}")

        print(f"\nNext steps:")
        print(f"  1. Build path cache (one-time, ~{stats['path_cache_estimate_min']:.0f} min):")
        print(f"     python -c \"from data.path_cache import build_training_cache; ...\"")
        if stats.get("chunked_evaluator_required"):
            print(f"  2. Use ChunkedEvaluator for evaluation (chunk_size='auto')")
        print(f"  3. Start training: python experiments/run_v2.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()