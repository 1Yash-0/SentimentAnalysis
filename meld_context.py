"""Small, standalone MELD text-context classifier.

Run from the project root: python meld_context.py --mode mixed
The module reads official CSVs and existing outputs/asr_{train,dev,test}.json caches.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

SENTIMENTS = ["negative", "neutral", "positive"]
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SEED = 42


def load_rows(root: Path, split: str) -> pd.DataFrame:
    """Load official rows, retaining only IDs with gold text and cached Whisper text."""
    frame = pd.read_csv(root / "data" / "MELD" / f"{split}.csv")
    frame["uid"] = "dia" + frame.Dialogue_ID.astype(str) + "_utt" + frame.Utterance_ID.astype(str)
    if frame.uid.duplicated().any(): raise ValueError(f"Duplicate official IDs in {split} CSV")
    cache_path = root / "outputs" / f"asr_{split}.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Required existing transcript cache is missing: {cache_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    frame["asr"] = frame.uid.map(cache)
    frame["Utterance"] = frame.Utterance.fillna("").astype(str).str.replace("\x92", "'", regex=False).str.replace("\x97", " ", regex=False)
    frame = frame[frame.asr.notna()].copy()
    frame["asr"] = frame.asr.fillna("").astype(str)
    frame["sent_y"] = frame.Sentiment.map({v: i for i, v in enumerate(SENTIMENTS)})
    frame["emo_y"] = frame.Emotion.map({v: i for i, v in enumerate(EMOTIONS)})
    if frame[["sent_y", "emo_y"]].isna().any().any():
        raise ValueError(f"Unexpected labels in {split} CSV")
    if frame.uid.duplicated().any() or frame.asr.isna().any(): raise ValueError(f"ID/cache alignment failed for {split}")
    return frame.reset_index(drop=True)


def make_contexts(frame: pd.DataFrame, current_only: bool = False) -> list[dict[str, str]]:
    """Build target-first contexts from up to 3 past and 2 future turns in-dialogue."""
    frame = frame.reset_index(drop=True)
    result: list[dict[str, str]] = [{} for _ in range(len(frame))]
    for _, group in frame.groupby("Dialogue_ID", sort=False):
        group = group.sort_values("Utterance_ID")
        speakers = {speaker: i + 1 for i, speaker in enumerate(group.Speaker.drop_duplicates())}
        records = list(group.iterrows())
        for pos, (index, row) in enumerate(records):
            if current_only:
                result[index] = {"gold": row.Utterance, "asr": row.asr}
                continue
            # Put the target first so truncation preserves the supervised utterance.
            selected = [records[pos]] + records[max(0, pos - 3):pos] + records[pos + 1:pos + 3]
            target_offset = 0
            result[index] = {
                "gold": " </s> ".join(f"<{('TARGET' if j == target_offset else 'OTHER')}_S{speakers[t.Speaker]}> {t.Utterance}" for j, (_, t) in enumerate(selected)),
                "asr": " </s> ".join(f"<{('TARGET' if j == target_offset else 'OTHER')}_S{speakers[t.Speaker]}> {t.asr}" for j, (_, t) in enumerate(selected)),
            }
    return result


class EncodedRows(Dataset):
    def __init__(self, contexts, labels, tokenizer, mode, max_len):
        self.labels, self.mode = labels, mode
        self.encoded = {source: tokenizer([row[source] for row in contexts], truncation=True, max_length=max_len) for source in ("gold", "asr")}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return i, self.labels[i]

    def features(self, source, index):
        # BatchEncoding[index] is input_ids only in some Transformers releases.
        return {key: values[index] for key, values in self.encoded[source].items()}


class ContextClassifier(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(0.1)
        self.sentiment = torch.nn.Linear(hidden, 3)
        self.emotion = torch.nn.Linear(hidden, 7)

    def forward(self, **inputs):
        pooled = self.dropout(self.encoder(**inputs).last_hidden_state[:, 0])
        return self.sentiment(pooled), self.emotion(pooled)


def _collator(tokenizer, encoded, training=False, epoch=0, source="gold"):
    def collate(batch):
        indices = [item[0] for item in batch]
        batch_source = source
        if training and encoded.mode == "mixed":
            batch_source = None
        if batch_source is None:
            # Stable per-row alternation, switching every epoch.
            sources = ["gold" if (i + epoch) % 2 == 0 else "asr" for i in indices]
            inputs = [encoded.features(s, i) for s, i in zip(sources, indices)]
            batch_inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        else:
            batch_inputs = tokenizer.pad([encoded.features(batch_source, i) for i in indices], padding=True, return_tensors="pt")
        labels = torch.tensor([encoded.labels[i] for i in indices], dtype=torch.long)
        return batch_inputs, labels
    return collate


def _probabilities(model, dataset, tokenizer, source, device, batch_size):
    model.eval()
    precision = amp_dtype(model, device)
    results = []
    collate = _collator(tokenizer, dataset, training=False, source=source)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=precision, enabled=device.type == "cuda"):
                sent, emo = model(**inputs)
            results.append((sent.float().softmax(-1).cpu().numpy(), emo.float().softmax(-1).cpu().numpy()))
    return np.concatenate([p[0] for p in results]), np.concatenate([p[1] for p in results])


def amp_dtype(model, device):
    if device.type == "cuda" and model.encoder.config.hidden_size >= 1024 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def save_predictions(frame, probabilities, split, transcript, out, model_tag):
    sent, emo = probabilities
    result = frame[["uid", "Dialogue_ID", "Utterance_ID", "Sentiment", "Emotion"]].copy()
    result.insert(0, "split", split)
    result.insert(1, "transcript_source", transcript)
    result["pred_sentiment"] = [SENTIMENTS[i] for i in sent.argmax(1)]
    result["pred_emotion"] = [EMOTIONS[i] for i in emo.argmax(1)]
    for i, label in enumerate(SENTIMENTS): result[f"sentiment_p_{label}"] = sent[:, i]
    for i, label in enumerate(EMOTIONS): result[f"emotion_p_{label}"] = emo[:, i]
    result.to_csv(out / f"predictions_{model_tag}_{split}_{transcript}.csv", index=False)


def predict_split(root: str | Path, checkpoint: str | Path, split="test", transcript="asr", batch_size=16):
    """Reload a saved checkpoint and return full probabilities aligned to CSV row IDs."""
    root, checkpoint = Path(root), Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_rows(root, split)
    tok = AutoTokenizer.from_pretrained(saved["model_name"])
    contexts = make_contexts(frame, current_only=bool(saved.get("current_only", False)))
    if transcript == "asr":
        contexts = [{"gold": row["asr"], "asr": row["asr"]} for row in contexts]
    elif transcript != "gold":
        raise ValueError("transcript must be gold or asr")
    ys = list(zip(frame.sent_y.astype(int), frame.emo_y.astype(int)))
    data = EncodedRows(contexts, ys, tok, "gold", saved["max_len"])
    model = ContextClassifier(saved["model_name"]).to(device)
    model.load_state_dict(saved["state_dict"])
    probs = _probabilities(model, data, tok, "gold", device, batch_size)
    save_predictions(frame, probs, split, transcript, root / "outputs", checkpoint.stem)
    return frame[["uid", "Dialogue_ID", "Utterance_ID"]].copy(), probs


def train(root: str | Path = ".", model_name="roberta-base", mode="mixed", epochs=5, batch_size=16, max_len=128, lr=2e-5, current_only=False):
    """Train using train labels; select checkpoints only by dev macro-F1."""
    if mode not in {"gold", "mixed"}: raise ValueError("mode must be gold or mixed")
    root, out = Path(root), Path(root) / "outputs"
    out.mkdir(exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, dev_df = load_rows(root, "train"), load_rows(root, "dev")
    tok = AutoTokenizer.from_pretrained(model_name)
    contexts = {name: make_contexts(df, current_only=current_only) for name, df in [("train", train_df), ("dev", dev_df)]}
    labels = {name: list(zip(df.sent_y.astype(int), df.emo_y.astype(int))) for name, df in [("train", train_df), ("dev", dev_df)]}
    data = {name: EncodedRows(contexts[name], labels[name], tok, mode, max_len) for name in contexts}
    model = ContextClassifier(model_name).to(device)
    if device.type == "cuda" and model.encoder.config.hidden_size >= 1024:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sy, ey = train_df.sent_y.to_numpy(), train_df.emo_y.to_numpy()
    w_s = torch.tensor(np.sqrt(compute_class_weight("balanced", classes=np.arange(3), y=sy)), dtype=torch.float32, device=device)
    w_e = torch.tensor(np.sqrt(compute_class_weight("balanced", classes=np.arange(7), y=ey)), dtype=torch.float32, device=device)
    loss_s, loss_e = torch.nn.CrossEntropyLoss(weight=w_s), torch.nn.CrossEntropyLoss(weight=w_e)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=False)
    steps = epochs * int(np.ceil(len(data["train"]) / batch_size))
    warm_steps = max(1, int(0.1 * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warm_steps, max(0.0, (steps - step) / max(1, steps - warm_steps))))
    precision = amp_dtype(model, device)
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == torch.float16)
    history, best = [], -1.0
    tag = f"context_{mode}_{Path(model_name).name}" + ("_current" if current_only else "")
    checkpoint = out / f"{tag}.pt"
    for epoch in range(epochs):
        model.train()
        updates = 0
        loader = DataLoader(data["train"], batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED + epoch), collate_fn=_collator(tok, data["train"], True, epoch))
        for step, (inputs, ys) in enumerate(loader, 1):
            inputs = {k: v.to(device) for k, v in inputs.items()}; ys = ys.to(device)
            with torch.autocast(device_type="cuda", dtype=precision, enabled=device.type == "cuda"):
                ps, pe = model(**inputs)
                loss = loss_s(ps.float(), ys[:, 0]) + loss_e(pe.float(), ys[:, 1])
            optimizer.zero_grad(set_to_none=True)
            amp.scale(loss).backward()
            if precision == torch.bfloat16:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scale_before = amp.get_scale(); amp.step(optimizer); amp.update()
            if amp.get_scale() >= scale_before:
                scheduler.step()
                updates += 1
            if step == 128 and updates == 0:
                raise RuntimeError("Mixed precision skipped the first 128 updates. Training is unstable.")
        sources = ["gold", "asr"] if mode == "mixed" else ["gold"]
        metrics = {}
        for source in sources:
            probs = _probabilities(model, data["dev"], tok, source, device, batch_size)
            sf = f1_score(dev_df.sent_y, probs[0].argmax(1), labels=range(3), average="macro", zero_division=0)
            ef = f1_score(dev_df.emo_y, probs[1].argmax(1), labels=range(7), average="weighted", zero_division=0)
            metrics[source] = (float(sf), float(ef))
        sf = float(np.mean([v[0] for v in metrics.values()])); ef = float(np.mean([v[1] for v in metrics.values()]))
        row = {"epoch": epoch + 1, "optimizer_updates": updates, "dev_sentiment_macro_f1": sf, "dev_emotion_weighted_f1": ef, "selection_score": float((sf + ef) / 2)}
        for source, values in metrics.items():
            row[f"dev_{source}_sentiment_macro_f1"], row[f"dev_{source}_emotion_weighted_f1"] = values
        history.append(row); print(row)
        if row["selection_score"] > best:
            best = row["selection_score"]
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({"state_dict": model.state_dict(), "model_name": model_name, "mode": mode, "max_len": max_len, "seed": SEED, "current_only": current_only}, temporary)
            temporary.replace(checkpoint)
    pd.DataFrame(history).to_csv(out / f"{tag}_history.csv", index=False)
    del optimizer, scheduler, amp
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    del saved
    test_df = load_rows(root, "test")  # Test data is read only after checkpoint selection is complete.
    contexts["test"] = make_contexts(test_df, current_only=current_only)
    labels["test"] = list(zip(test_df.sent_y.astype(int), test_df.emo_y.astype(int)))
    data["test"] = EncodedRows(contexts["test"], labels["test"], tok, mode, max_len)
    for split, df in [("dev", dev_df), ("test", test_df)]:
        sources = ["gold", "asr"] if mode == "mixed" else ["gold"]
        for source in sources:
            chosen = contexts[split] if source == "gold" else [{"gold": c["asr"], "asr": c["asr"]} for c in contexts[split]]
            eval_data = EncodedRows(chosen, labels[split], tok, "gold", max_len)
            if len(df) != len(eval_data) or len(df.uid.unique()) != len(df): raise ValueError(f"Prediction IDs misaligned for {split}")
            save_predictions(df, _probabilities(model, eval_data, tok, "gold", device, batch_size), split, source, out, tag)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="."); parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--mode", choices=["gold", "mixed"], default="mixed")
    parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--current-only", action="store_true", help="Ablation using only the target utterance, without context or speaker markers")
    args = parser.parse_args()
    print("Saved best development-selected model:", train(args.root, args.model_name, args.mode, args.epochs, args.batch_size, args.max_len, lr=args.lr, current_only=args.current_only))


if __name__ == "__main__": main()
