"""Select and score the final MELD classifiers using saved predictions."""

import json
import re
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from threadpoolctl import threadpool_limits

from meld_context import EMOTIONS, SENTIMENTS, load_rows
from meld_summary import evaluate_trajectories, summarize_dialogue


TASKS = [
    ("sentiment", "sent", "Sentiment", SENTIMENTS),
    ("emotion", "emo", "Emotion", EMOTIONS),
]
TEXT_TRIALS = [
    "context_gold_roberta-base",
    "context_mixed_roberta-base",
    "context_mixed_roberta-base_current",
    "context_mixed_roberta-base-go_emotions",
    "context_mixed_roberta-large",
]


def aligned_csv(path, rows):
    predictions = pd.read_csv(path).set_index("uid", verify_integrity=True)
    if set(predictions.index) != set(rows.uid):
        raise ValueError(f"Utterance IDs do not match in {path.name}")
    predictions = predictions.loc[rows.uid]
    for column in ["Sentiment", "Emotion"]:
        if not np.array_equal(predictions[column], rows[column]):
            raise ValueError(f"{column} labels do not match in {path.name}")
    return predictions


def metrics(truth, probabilities, labels):
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": 100 * accuracy_score(truth, predicted),
        "macro_f1": 100 * f1_score(truth, predicted, labels=range(len(labels)), average="macro", zero_division=0),
        "weighted_f1": 100 * f1_score(truth, predicted, labels=range(len(labels)), average="weighted", zero_division=0),
    }


def normalize_transcript(text):
    return re.sub(r"[^a-z0-9' ]", " ", text.lower())


def evaluate(root="."):
    root = Path(root)
    out = root / "outputs"
    rows = {split: load_rows(root, split) for split in ["train", "dev", "test"]}
    if [len(rows[split]) for split in ["train", "dev", "test"]] != [9988, 1108, 2610]:
        raise ValueError("The final evaluation requires all available official MELD rows")
    old = {split: aligned_csv(out / f"predictions_{split}.csv", rows[split]) for split in ["dev", "test"]}
    truth = {(split, task): frame[column].map(labels.index).to_numpy()
             for split, frame in rows.items() for task, _, column, labels in TASKS}
    results, classes, development, prediction_rows, trajectories = [], [], [], [], []
    settings = {
        "description": "RoBERTa sees the target utterance and up to five nearby turns, with dialogue-local speaker markers.",
        "whisper_test_wer": round(100 * jiwer.wer(
            rows["test"].Utterance.map(normalize_transcript).tolist(),
            rows["test"].asr.map(normalize_transcript).tolist(),
        ), 2),
        "selected_models": {}, "weights": {},
    }
    examples = {}
    for input_name, transcript, column, old_key in [
        ("human", "gold", "Utterance", "text_gold"), ("whisper", "asr", "asr", "text_asr"),
    ]:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        train_words = vectorizer.fit_transform(rows["train"][column])
        with threadpool_limits(limits=4):
            tfidf = {
                task: LogisticRegression(max_iter=1000, class_weight="balanced").fit(train_words, truth["train", task])
                for task, _, _, _ in TASKS
            }
        words = {split: vectorizer.transform(rows[split][column]) for split in ["dev", "test"]}
        candidates = {"Utterance RoBERTa": old["dev"]}
        for name in TEXT_TRIALS:
            path = out / f"predictions_{name}_dev_{transcript}.csv"
            test_path = out / f"predictions_{name}_test_{transcript}.csv"
            if path.is_file() and test_path.is_file():
                candidates[name] = aligned_csv(path, rows["dev"])

        def text_probs(frame, name, task, short, labels):
            prefix = f"{old_key}_{short}_prob_" if name == "Utterance RoBERTa" else f"{task}_p_"
            return frame[[prefix + label for label in labels]].to_numpy()

        scores = {}
        for name, frame in candidates.items():
            task_scores = []
            for task, short, _, labels in TASKS:
                score = metrics(truth["dev", task], text_probs(frame, name, task, short, labels), labels)
                task_scores.append(score["macro_f1" if task == "sentiment" else "weighted_f1"])
                development.append({"input": input_name, "model": name, "task": task, **score})
            scores[name] = float(np.mean(task_scores))
        selected = max(scores, key=scores.get)
        settings["selected_models"][input_name] = selected
        settings["weights"][input_name] = {}
        final_frames = {"dev": candidates[selected], "test": old["test"]}
        if selected != "Utterance RoBERTa":
            final_frames["test"] = aligned_csv(
                out / f"predictions_{selected}_test_{transcript}.csv", rows["test"],
            )
        predictions = rows["test"][["uid", "Dialogue_ID", "Utterance_ID", "Speaker", "Sentiment", "Emotion"]].copy()
        predictions.insert(0, "input", input_name)
        for task, short, _, labels in TASKS:
            metric = "macro_f1" if task == "sentiment" else "weighted_f1"
            dev_text = text_probs(final_frames["dev"], selected, task, short, labels)
            audio_columns = [f"speech_{short}_prob_{label}" for label in labels]
            dev_audio = old["dev"][audio_columns].to_numpy()
            weight = max(
                (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                key=lambda w: metrics(truth["dev", task], (1-w)*dev_text + w*dev_audio, labels)[metric],
            )
            settings["weights"][input_name][task] = {"text": round(1-weight, 2), "audio": weight}
            prior = np.bincount(truth["train", task], minlength=len(labels))
            prior = prior / prior.sum()
            for split in ["dev", "test"]:
                text = text_probs(final_frames[split], selected, task, short, labels)
                audio = old[split][audio_columns].to_numpy()
                models = {
                    "Majority": np.tile(prior, (len(rows[split]), 1)),
                    "TF-IDF": tfidf[task].predict_proba(words[split]),
                    "Utterance RoBERTa": text_probs(old[split], "Utterance RoBERTa", task, short, labels),
                    "Audio": audio,
                    "Selected text": text,
                    "Selected text + audio": (1-weight)*text + weight*audio,
                }
                if input_name == "whisper":
                    columns = [f"fuse_avg_{short}_prob_{label}" for label in labels]
                    models["Previous fusion"] = old[split][columns].to_numpy()
                for name, probs in models.items():
                    y = truth[split, task]
                    results.append({"split": split, "input": input_name, "model": name, "task": task,
                                    **metrics(y, probs, labels)})
                    if split == "test" and name in ["Utterance RoBERTa", "Selected text", "Selected text + audio"]:
                        report = classification_report(y, probs.argmax(1), labels=range(len(labels)),
                                                       target_names=labels, output_dict=True, zero_division=0)
                        for label in labels:
                            classes.append({"input": input_name, "model": name, "task": task,
                                            "label": label, **report[label]})
                if split == "test":
                    predictions[f"baseline_{task}"] = np.asarray(labels)[models["Utterance RoBERTa"].argmax(1)]
                    predictions[f"text_{task}"] = np.asarray(labels)[text.argmax(1)]
                    predictions[f"final_{task}"] = np.asarray(labels)[models["Selected text + audio"].argmax(1)]
        prediction_rows.append(predictions)
        summary_input = predictions.assign(text=rows["test"][column])
        for dialogue_id, dialogue in summary_input.groupby("Dialogue_ID"):
            for method in ["baseline_emotion", "final_emotion"]:
                score = evaluate_trajectories(dialogue, method, "Emotion")
                trajectories.append({"input": input_name, "dialogue": dialogue_id, "method": method,
                                     "trajectory_similarity": score["similarity"]})
            if input_name not in examples and len(dialogue) >= 8 and dialogue.Speaker.nunique() >= 3:
                examples[input_name] = {"dialogue": int(dialogue_id),
                                        **summarize_dialogue(dialogue, "text", "final_emotion")}

    # Measure transcript quality with the Whisper-selected checkpoint held fixed.
    shared = settings["selected_models"]["whisper"]
    settings["matched_input_model"] = shared
    for input_name, transcript, old_key in [("human", "gold", "text_gold"), ("whisper", "asr", "text_asr")]:
        for split in ["dev", "test"]:
            frame = old[split] if shared == "Utterance RoBERTa" else aligned_csv(
                out / f"predictions_{shared}_{split}_{transcript}.csv", rows[split],
            )
            for task, short, _, labels in TASKS:
                prefix = f"{old_key}_{short}_prob_" if shared == "Utterance RoBERTa" else f"{task}_p_"
                probs = frame[[prefix + label for label in labels]].to_numpy()
                results.append({"split": split, "input": input_name, "model": "Same checkpoint",
                                "task": task, **metrics(truth[split, task], probs, labels)})

    for input_name, transcript in [("human", "gold"), ("whisper", "asr")]:
        for name, tag in [("Mixed base, target only", "_current"), ("Mixed base, nearby turns", "")]:
            frame = aligned_csv(out / f"predictions_context_mixed_roberta-base{tag}_test_{transcript}.csv",
                                rows["test"])
            for task, _, _, labels in TASKS:
                probs = frame[[f"{task}_p_{label}" for label in labels]].to_numpy()
                results.append({"split": "test", "input": input_name, "model": name, "task": task,
                                **metrics(truth["test", task], probs, labels)})

    table = pd.DataFrame(results).round(2)
    table.to_csv(out / "final_results.csv", index=False)
    pd.DataFrame(classes).to_csv(out / "final_per_class.csv", index=False)
    pd.DataFrame(development).round(2).to_csv(out / "final_dev_selection.csv", index=False)
    pd.concat(prediction_rows).to_csv(out / "final_predictions.csv", index=False)
    pd.DataFrame(trajectories).to_csv(out / "final_summary_scores.csv", index=False)
    (out / "final_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out / "final_summary_examples.json").write_text(
        json.dumps(examples, indent=2, default=lambda value: value.item()), encoding="utf-8",
    )
    return table


if __name__ == "__main__":
    print(evaluate().query("split == 'test'").to_string(index=False))
