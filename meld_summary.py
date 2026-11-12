"""Small, deterministic summaries for MELD-style dialogue frames."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _check_columns(frame, *columns):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _labels(frame, column, speaker_col="Speaker"):
    result = {}
    for speaker, group in frame.groupby(speaker_col, sort=False, dropna=True):
        labels = []
        for value in group[column]:
            if pd.isna(value) or not str(value).strip():
                continue
            value = str(value).strip()
            if not labels or labels[-1] != value:
                labels.append(value)
        if labels:
            result[speaker] = labels
    return result


def summarize_dialogue(frame, text_col, emotion_col, speaker_col="Speaker",
                       order_col="Utterance_ID"):
    """Return representative original quotes and per-speaker emotion runs."""
    _check_columns(frame, speaker_col, order_col, text_col, emotion_col)
    ordered = frame.sort_values(order_col, kind="stable", na_position="last")
    valid = ordered[ordered[text_col].notna()].copy()
    valid[text_col] = valid[text_col].map(str)
    valid = valid[valid[text_col].str.strip().ne("")]
    quotes = []
    if not valid.empty:
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform(valid[text_col])
            center = np.asarray(matrix.mean(axis=0))
            relevance, redundancy = cosine_similarity(matrix, center).ravel(), cosine_similarity(matrix)
        except ValueError:
            relevance, redundancy = np.zeros(len(valid)), np.zeros((len(valid), len(valid)))
        remaining, chosen = list(range(len(valid))), []
        for _ in range(min(3, len(remaining))):
            scores = [(relevance[i] - 0.35 * max((redundancy[i, j] for j in chosen), default=0), -i, i)
                      for i in remaining]
            selected = max(scores)[2]
            chosen.append(selected)
            remaining.remove(selected)
        for i in sorted(chosen):
            row = valid.iloc[i]
            quotes.append({"speaker": row[speaker_col], "utterance_id": row[order_col],
                           "text": row[text_col]})
    trajectories = _labels(ordered, emotion_col, speaker_col)
    return {"quotes": quotes, "trajectories": trajectories}


def evaluate_trajectories(frame, predicted_col, human_col, speaker_col="Speaker",
                          order_col="Utterance_ID"):
    """Score ordered, repeat-collapsed trajectories with per-speaker LCS similarity."""
    _check_columns(frame, speaker_col, order_col, predicted_col, human_col)
    ordered = frame.sort_values(order_col, kind="stable", na_position="last")
    per_speaker = {}
    for speaker, group in ordered.groupby(speaker_col, sort=False, dropna=True):
        predicted = _labels(group, predicted_col, speaker_col).get(speaker, [])
        human = _labels(group, human_col, speaker_col).get(speaker, [])
        if not predicted or not human:
            continue
        row = [0] * (len(human) + 1)
        for p in predicted:
            next_row = [0]
            for j, h in enumerate(human, 1):
                next_row.append(row[j - 1] + 1 if p == h else max(row[j], next_row[-1]))
            row = next_row
        per_speaker[speaker] = row[-1] / max(len(predicted), len(human))
    return {"similarity": float(np.mean(list(per_speaker.values()))) if per_speaker else None,
            "per_speaker": per_speaker}
