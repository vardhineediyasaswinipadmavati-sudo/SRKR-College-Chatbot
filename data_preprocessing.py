import json
import re
import string
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split

import config

def clean_text(text: str) -> str:
    """Normalize and clean raw text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Remove excess whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into word tokens."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()

class SRKRDataPreprocessor:
    def __init__(self, max_vocab_size: int = config.MAX_VOCAB_SIZE, max_len: int = config.MAX_SEQUENCE_LENGTH):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.tag2idx: Dict[str, int] = {}
        self.idx2tag: Dict[int, str] = {}
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, Any] = {}

    def load_dataset(self, filepath: Path) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        Dynamically loads and validates the SRKR JSON dataset.
        Adapts to schema whether it's 'intents', 'qa_pairs', or 'knowledge_base'.
        Preserves original data, validates non-empty entries, and filters exact duplicates.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found at: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_intents = []
        if isinstance(data, dict):
            if "intents" in data and isinstance(data["intents"], list):
                raw_intents = data["intents"]
            elif "data" in data and isinstance(data["data"], list):
                raw_intents = data["data"]
            else:
                # If dictionary of tag -> {patterns, responses}
                for k, v in data.items():
                    if isinstance(v, dict) and ("patterns" in v or "responses" in v):
                        item = {"tag": k}
                        item.update(v)
                        raw_intents.append(item)
        elif isinstance(data, list):
            raw_intents = data

        if not raw_intents:
            raise ValueError(f"Could not extract intent records from dataset at {filepath}")

        patterns_list = []
        labels_list = []
        knowledge_base = {}

        seen_patterns = set()
        duplicates_removed = 0
        empty_records_removed = 0

        for item in raw_intents:
            tag = item.get("tag") or item.get("intent") or item.get("label")
            if not tag:
                empty_records_removed += 1
                continue
            
            tag = str(tag).strip()
            patterns = item.get("patterns") or item.get("questions") or []
            responses = item.get("responses") or item.get("answers") or []
            category = item.get("category", "General")

            if not patterns or not responses:
                empty_records_removed += 1
                continue

            cleaned_responses = [str(r).strip() for r in responses if str(r).strip()]
            if not cleaned_responses:
                empty_records_removed += 1
                continue

            # Store in knowledge base
            knowledge_base[tag] = {
                "tag": tag,
                "category": category,
                "responses": cleaned_responses,
                "patterns_sample": patterns[:3]
            }

            for p in patterns:
                p_str = str(p).strip()
                if not p_str:
                    continue
                # Normalize for deduplication check
                p_norm = clean_text(p_str)
                if not p_norm:
                    continue
                if p_norm in seen_patterns:
                    duplicates_removed += 1
                    continue
                seen_patterns.add(p_norm)
                patterns_list.append(p_str)
                labels_list.append(tag)

        self.knowledge_base = knowledge_base

        # Check and independently preserve separate knowledge base if present
        separate_kb = {}
        if isinstance(data, dict) and "knowledge_base" in data:
            raw_kb = data["knowledge_base"]
            if isinstance(raw_kb, list):
                for kb_item in raw_kb:
                    kb_id = kb_item.get("id") or kb_item.get("tag")
                    if kb_id:
                        separate_kb[str(kb_id)] = kb_item
            elif isinstance(raw_kb, dict):
                separate_kb = raw_kb
        self.separate_knowledge_base = separate_kb

        self.stats = {
            "total_intents": len(knowledge_base),
            "total_patterns": len(patterns_list),
            "total_responses": sum(len(v["responses"]) for v in knowledge_base.values()),
            "separate_kb_records": len(separate_kb),
            "duplicates_removed": duplicates_removed,
            "empty_records_removed": empty_records_removed
        }

        print("=" * 60)
        print("SRKR DATASET INSPECTION & PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Unique Intents Found:       {self.stats['total_intents']}")
        print(f"Total Cleaned Patterns/Questions: {self.stats['total_patterns']}")
        print(f"Total Verified Responses/Answers: {self.stats['total_responses']}")
        print(f"Independent Knowledge Base Items: {self.stats['separate_kb_records']}")
        print(f"Exact Duplicate Patterns Removed: {self.stats['duplicates_removed']}")
        print(f"Empty/Invalid Records Filtered:   {self.stats['empty_records_removed']}")
        print("=" * 60)

        return patterns_list, labels_list, knowledge_base

    def fit(self, patterns: List[str], labels: List[str]):
        """Build vocabulary and label encoders."""
        # 1. Build Label Encoders
        unique_tags = sorted(list(set(labels)))
        self.tag2idx = {tag: i for i, tag in enumerate(unique_tags)}
        self.idx2tag = {i: tag for i, tag in enumerate(unique_tags)}

        # 2. Build Vocabulary
        word_freq = {}
        for text in patterns:
            tokens = tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        # Limit to max_vocab_size - 2 (<PAD>, <UNK>)
        top_words = sorted_words[: self.max_vocab_size - 2]

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for i, (word, _) in enumerate(top_words, start=2):
            self.word2idx[word] = i
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.stats["vocabulary_size"] = len(self.word2idx)
        print(f"Vocabulary Size (with <PAD>, <UNK>): {self.stats['vocabulary_size']}")

    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert list of texts to padded integer sequences."""
        sequences = []
        for text in texts:
            tokens = tokenize(text)
            seq = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
            # Truncate or pad
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            else:
                seq = seq + [self.word2idx["<PAD>"]] * (self.max_len - len(seq))
            sequences.append(seq)
        return np.array(sequences, dtype=np.int32)

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Convert list of string tags to integer labels."""
        return np.array([self.tag2idx[label] for label in labels], dtype=np.int32)

    def prepare_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        raw_patterns: List[str],
        test_size: float = config.TEST_SPLIT,
        val_size: float = config.VAL_SPLIT,
        random_state: int = config.RANDOM_STATE
    ) -> Dict[str, Any]:
        """
        Split dataset into train, validation, and test partitions.
        Uses stratified splitting when class sizes and partition sizes allow;
        gracefully falls back to safe random splitting if partition size is
        less than the number of classes (as required by scikit-learn).
        """
        indices = np.arange(len(y))
        num_classes = len(np.unique(y))
        test_count = int(np.ceil(len(y) * test_size))
        unique_classes, counts = np.unique(y, return_counts=True)
        min_count = int(np.min(counts))

        can_stratify_test = (test_count >= num_classes) and (min_count >= 2)
        stratify_test = y if can_stratify_test else None

        if not can_stratify_test:
            print(
                f"[Dataset Split Info] Stratified split disabled: test sample count ({test_count}) "
                f"is smaller than the number of intent classes ({num_classes}) or min class count ({min_count}) < 2. "
                f"Using safe reproducible random splitting."
            )
        else:
            print(f"[Dataset Split Info] Stratified test split enabled (minimum class size: {min_count}).")

        # First split into train_val and test
        idx_train_val, idx_test, y_train_val, y_test = train_test_split(
            indices, y, test_size=test_size, random_state=random_state, stratify=stratify_test
        )

        # Then split train_val into train and val
        val_count = int(np.ceil(len(idx_train_val) * (val_size / (1.0 - test_size))))
        unique_train_val, counts_train_val = np.unique(y_train_val, return_counts=True)
        min_count_tv = int(np.min(counts_train_val))
        can_stratify_val = (val_count >= len(unique_train_val)) and (min_count_tv >= 2)
        stratify_val = y_train_val if can_stratify_val else None

        val_relative_size = val_size / (1.0 - test_size)
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=stratify_val
        )

        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]
        X_test, y_test = X[idx_test], y[idx_test]

        patterns_train = [raw_patterns[i] for i in idx_train]
        patterns_val = [raw_patterns[i] for i in idx_val]
        patterns_test = [raw_patterns[i] for i in idx_test]

        self.stats["train_samples"] = int(len(y_train))
        self.stats["val_samples"] = int(len(y_val))
        self.stats["test_samples"] = int(len(y_test))

        print(f"Data Splits -> Train: {len(y_train)}, Validation: {len(y_val)}, Test: {len(y_test)}")

        return {
            "X_train": X_train, "y_train": y_train, "patterns_train": patterns_train,
            "X_val": X_val, "y_val": y_val, "patterns_val": patterns_val,
            "X_test": X_test, "y_test": y_test, "patterns_test": patterns_test,
        }

    def save_artifacts(self, output_dir: Path):
        """Save preprocessed artifacts, encoders, and knowledge base."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata & encoders
        metadata = {
            "max_vocab_size": self.max_vocab_size,
            "max_len": self.max_len,
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "tag2idx": self.tag2idx,
            "idx2tag": {str(k): v for k, v in self.idx2tag.items()},
            "stats": self.stats
        }
        with open(output_dir / "preprocessor_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(output_dir / "knowledge_base.json", "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, indent=2)

        if hasattr(self, "separate_knowledge_base") and self.separate_knowledge_base:
            with open(output_dir / "separate_knowledge_base.json", "w", encoding="utf-8") as f:
                json.dump(self.separate_knowledge_base, f, indent=2)

        with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

        print(f"Preprocessed artifacts successfully saved to: {output_dir}")

    def load_artifacts(self, input_dir: Path):
        """Load saved preprocessor metadata and knowledge base."""
        with open(input_dir / "preprocessor_config.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.max_vocab_size = metadata["max_vocab_size"]
        self.max_len = metadata["max_len"]
        self.word2idx = metadata["word2idx"]
        self.idx2word = {int(k): v for k, v in metadata["idx2word"].items()}
        self.tag2idx = metadata["tag2idx"]
        self.idx2tag = {int(k): v for k, v in metadata["idx2tag"].items()}
        self.stats = metadata.get("stats", {})

        with open(input_dir / "knowledge_base.json", "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

def run_pipeline() -> Tuple[SRKRDataPreprocessor, Dict[str, Any]]:
    """Execute complete preprocessing pipeline."""
    preprocessor = SRKRDataPreprocessor(
        max_vocab_size=config.MAX_VOCAB_SIZE,
        max_len=config.MAX_SEQUENCE_LENGTH
    )
    patterns, labels, _ = preprocessor.load_dataset(config.ORIGINAL_DATASET_PATH)
    preprocessor.fit(patterns, labels)

    X = preprocessor.texts_to_sequences(patterns)
    y = preprocessor.encode_labels(labels)

    splits = preprocessor.prepare_splits(X, y, patterns)
    preprocessor.save_artifacts(config.PROCESSED_DATA_DIR)

    # Save splits as npz
    np.savez_compressed(
        config.PROCESSED_DATA_DIR / "dataset_splits.npz",
        X_train=splits["X_train"], y_train=splits["y_train"],
        X_val=splits["X_val"], y_val=splits["y_val"],
        X_test=splits["X_test"], y_test=splits["y_test"]
    )
    with open(config.PROCESSED_DATA_DIR / "patterns_splits.json", "w", encoding="utf-8") as f:
        json.dump({
            "patterns_train": splits["patterns_train"],
            "patterns_val": splits["patterns_val"],
            "patterns_test": splits["patterns_test"]
        }, f, indent=2)

    return preprocessor, splits

if __name__ == "__main__":
    run_pipeline()
