import nbformat
from pathlib import Path

cells = []

def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip()))

def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip()))

md(r"""
# Prepare MELD inputs and baselines

Prepare data, local Whisper transcripts, and baseline predictions for MELD_Final.ipynb.
""")

md(r"""
## 1. Setup
""")

code(r"""
import os, gc, json, time, random
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
import torch
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import logging
logging.getLogger('httpx').setLevel(logging.WARNING)
if os.name == 'nt':
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.add_dll_directory(torch_lib)
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path.cwd()
DATA = ROOT / 'data' / 'MELD'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
print('device:', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')
RETRAIN_TEXT = False
SENTIMENTS = ['negative', 'neutral', 'positive']
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
""")

md(r"""
## 2. MELD data
""")

code(r"""
import urllib.request, tarfile
HF = 'https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/'
(DATA / 'archive').mkdir(parents=True, exist_ok=True)
for split in ['train', 'dev', 'test']:
    csv = DATA / f'{split}.csv'
    if not csv.exists():
        urllib.request.urlretrieve(HF + f'{split}.csv', csv)
    if not (DATA / split).exists():
        tgz = DATA / 'archive' / f'{split}.tar.gz'
        if not tgz.exists():
            print('downloading', tgz.name)
            urllib.request.urlretrieve(HF + f'archive/{split}.tar.gz', tgz)
        with tarfile.open(tgz) as t:
            t.extractall(DATA, filter='data')

def load_split(split):
    df = pd.read_csv(DATA / f'{split}.csv')
    df['split'] = split
    df['uid'] = 'dia' + df.Dialogue_ID.astype(str) + '_utt' + df.Utterance_ID.astype(str)
    df['path'] = df.uid.map(lambda u: str(DATA / split / f'{u}.flac'))
    df = df[df.path.map(os.path.exists)].reset_index(drop=True)
    df['Utterance'] = df.Utterance.str.replace('\x92', "'", regex=False).str.replace('\x97', ' ', regex=False)
    df['sent_y'] = df.Sentiment.map(SENTIMENTS.index)
    df['emo_y'] = df.Emotion.map(EMOTIONS.index)
    return df

train = load_split('train')
dev = load_split('dev')
test = load_split('test')
print({'train': len(train), 'dev': len(dev), 'test': len(test)})
""")

md(r"""
## 3. Whisper large-v3-turbo transcripts
""")

code(r"""
def load_audio(path):
    a, sr = sf.read(path, dtype='float32')
    if a.ndim > 1:
        a = a.mean(1)
    assert sr == 16000, sr
    return a

ASR_MODEL = 'large-v3-turbo'

def add_transcripts(df, name):
    cache = OUT / f'asr_{name}.json'
    transcripts = json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else {}
    missing = df[~df.uid.isin(transcripts)]
    if len(missing):
        from faster_whisper import WhisperModel
        asr_model = WhisperModel(ASR_MODEL, device=DEVICE, compute_type='float16' if DEVICE == 'cuda' else 'int8')
        for i, row in enumerate(missing.itertuples(), 1):
            segments, _ = asr_model.transcribe(load_audio(row.path), language='en', beam_size=5)
            transcripts[row.uid] = ' '.join((segment.text.strip() for segment in segments))
            if i % 100 == 0 or i == len(missing):
                temporary_cache = cache.with_suffix('.tmp')
                temporary_cache.write_text(json.dumps(transcripts, ensure_ascii=False), encoding='utf-8')
                temporary_cache.replace(cache)
                print(f'{name}: transcribed {i:,}/{len(missing):,} missing clips')
        del asr_model
        torch.cuda.empty_cache()
    df['asr'] = df.uid.map(transcripts)
    assert df.asr.notna().all()

for df, name in [(train, 'train'), (dev, 'dev'), (test, 'test')]:
    add_transcripts(df, name)
""")

md(r"""
## 4. Audio classifier
""")

code(r"""
E2V_TO_MELD = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 6: 'sadness', 7: 'surprise'}

def e2v_features(df, name):
    cache = OUT / f'e2v_{name}.npz'
    store = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        store = {u: (e, p) for u, e, p in zip(z['uid'], z['emb'], z['prob'])}
    todo = [(u, p) for u, p in zip(df.uid, df.path) if u not in store]
    if todo:
        from funasr import AutoModel
        e2v = AutoModel(model='emotion2vec/emotion2vec_plus_large', hub='hf', disable_update=True, device=DEVICE)
        MAX_SEC, t = (30, time.time())
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
            res = e2v.generate([load_audio(p)[:MAX_SEC * 16000] for _, p in chunk], granularity='utterance', extract_embedding=True, disable_pbar=True)
            for (u, _), r in zip(chunk, res):
                s = np.array(r['scores'])
                pr = np.array([s[k] for k in E2V_TO_MELD])
                store[u] = (r['feats'].astype(np.float32), (pr / max(pr.sum(), 1e-08)).astype(np.float32))
        print(f'{name}: {len(todo)} clips encoded in {time.time() - t:.0f}s')
        uids = list(store)
        np.savez(cache, uid=np.array(uids), emb=np.stack([store[u][0] for u in uids]), prob=np.stack([store[u][1] for u in uids]))
        del e2v
        gc.collect()
        torch.cuda.empty_cache()
    return (np.stack([store[u][0] for u in df.uid]), np.stack([store[u][1] for u in df.uid]))
S = {n: e2v_features(d, n) for n, d in [('train', train), ('dev', dev), ('test', test)]}
print({k: v[0].shape for k, v in S.items()})
""")

code(r"""
scaler = StandardScaler().fit(S['train'][0])
speech_heads = {}
for task, ycol in [('sent', 'sent_y'), ('emo', 'emo_y')]:
    candidates, scores = ([], [])
    for c_value in [0.1, 0.5, 1.0]:
        clf = LogisticRegression(max_iter=1000, C=c_value, class_weight='balanced', random_state=SEED)
        clf.fit(scaler.transform(S['train'][0]), train[ycol])
        pred = clf.predict(scaler.transform(S['dev'][0]))
        score = f1_score(dev[ycol], pred, labels=np.arange(len(clf.classes_)), average='macro', zero_division=0)
        candidates.append(clf)
        scores.append({'C': c_value, 'dev_macro_f1': float(score)})
    best_i = max(range(len(scores)), key=lambda i: (scores[i]['dev_macro_f1'], -scores[i]['C']))
    speech_heads[task] = candidates[best_i]
P = {}
for n in ['dev', 'test']:
    features = scaler.transform(S[n][0])
    for task in ['sent', 'emo']:
        P['speech', n, task] = speech_heads[task].predict_proba(features)
""")

md(r"""
## 5. Utterance-only RoBERTa baseline
""")

code(r"""
from transformers import AutoTokenizer, AutoModel
from sklearn.utils.class_weight import compute_class_weight
TXT_MODEL, MAXLEN, BS, EPOCHS, LR = ('roberta-base', 64, 32, 3, 2e-05)
tok = AutoTokenizer.from_pretrained(TXT_MODEL)

class TextClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(TXT_MODEL)
        h = self.enc.config.hidden_size
        self.drop = torch.nn.Dropout(0.1)
        self.sent, self.emo = (torch.nn.Linear(h, 3), torch.nn.Linear(h, 7))

    def forward(self, **x):
        cls = self.drop(self.enc(**x).last_hidden_state[:, 0])
        return (self.sent(cls), self.emo(cls))

def batches(texts, ys=None, shuffle=False):
    idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt').to(DEVICE)
        yield (enc, None if ys is None else [torch.tensor(y[b], device=DEVICE) for y in ys])

@torch.no_grad()
def predict_text(model, texts):
    model.eval()
    ps, pe = ([], [])
    for enc, _ in batches(list(texts)):
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == 'cuda'):
            ls, le = model(**enc)
        ps.append(ls.float().softmax(-1).cpu().numpy())
        pe.append(le.float().softmax(-1).cpu().numpy())
    return (np.concatenate(ps), np.concatenate(pe))

def train_text(model, texts, checkpoint, history_file, dev_texts=None):
    if dev_texts is None:
        dev_texts = dev.asr.replace('', '...')
    cw = lambda y, k: torch.tensor(np.sqrt(compute_class_weight('balanced', classes=np.arange(k), y=y)), dtype=torch.float, device=DEVICE)
    loss_s, loss_e = (torch.nn.CrossEntropyLoss(weight=cw(train.sent_y, 3)), torch.nn.CrossEntropyLoss(weight=cw(train.emo_y, 7)))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = EPOCHS * int(np.ceil(len(train) / BS))
    warm = int(0.1 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))
    scaler_amp = torch.amp.GradScaler(enabled=DEVICE == 'cuda')
    ys = [train.sent_y.values, train.emo_y.values]
    best = -1
    history = []
    temporary_checkpoint = checkpoint.with_suffix('.tmp')
    for ep in range(EPOCHS):
        model.train()
        t = time.time()
        for enc, (ysent, yemo) in batches(texts, ys, shuffle=True):
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == 'cuda'):
                ls, le = model(**enc)
                loss = loss_s(ls.float(), ysent) + loss_e(le.float(), yemo)
            opt.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()
            sched.step()
        ps, pe = predict_text(model, dev_texts)
        sentiment_f1 = f1_score(dev.sent_y, ps.argmax(1), labels=range(3), average='macro', zero_division=0)
        emotion_f1 = f1_score(dev.emo_y, pe.argmax(1), labels=range(7), average='macro', zero_division=0)
        f1 = (sentiment_f1 + emotion_f1) / 2
        history.append({'epoch': ep + 1, 'sentiment_macro_f1': sentiment_f1, 'emotion_macro_f1': emotion_f1, 'mean_macro_f1': f1})
        print(f'epoch {ep + 1}: dev mean macro-F1 = {f1:.4f}  ({time.time() - t:.0f}s)')
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint)
    pd.DataFrame(history).to_csv(history_file, index=False)

ckpt = OUT / 'roberta_meld_balanced.pt'
model = TextClassifier().to(DEVICE)
if not ckpt.exists() or RETRAIN_TEXT:
    train_text(model, list(train.Utterance), ckpt, OUT / 'training_history.csv')
model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
for n, df in [('dev', dev), ('test', test)]:
    for src, col in [('text_gold', 'Utterance'), ('text_asr', 'asr')]:
        P[src, n, 'sent'], P[src, n, 'emo'] = predict_text(model, df[col].replace('', '...'))
del model
gc.collect()
torch.cuda.empty_cache()
""")

md(r"""
## 6. Save the baseline predictions
""")

code(r"""
for task, column in [('sent', 'sent_y'), ('emo', 'emo_y')]:
    text, audio = P['text_asr', 'dev', task], P['speech', 'dev', task]
    classes = np.arange(text.shape[1])
    scores = []
    for weight in np.linspace(0, 1, 21):
        prediction = (weight * text + (1 - weight) * audio).argmax(1)
        score = f1_score(dev[column], prediction, labels=classes, average='macro', zero_division=0)
        scores.append((score, float(weight)))
    weight = max(scores, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]
    for split in ['dev', 'test']:
        text, audio = P['text_asr', split, task], P['speech', split, task]
        P['fuse_avg', split, task] = weight * text + (1 - weight) * audio

for split, frame in [('dev', dev), ('test', test)]:
    predictions = frame[['uid', 'Dialogue_ID', 'Utterance_ID', 'Sentiment', 'Emotion']].copy()
    for model in ['text_gold', 'text_asr', 'speech', 'fuse_avg']:
        for task, names in [('sent', SENTIMENTS), ('emo', EMOTIONS)]:
            probabilities = P[model, split, task]
            for i, name in enumerate(names):
                predictions[f'{model}_{task}_prob_{name}'] = probabilities[:, i]
    predictions.to_csv(OUT / f'predictions_{split}.csv', index=False)
    print(f'Saved {len(predictions)} {split} baseline predictions')
""")

notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "sentiment-nlp", "display_name": "Python (SentimentAnalysisNLP)", "language": "python"}})
nbformat.write(notebook, Path(__file__).with_name("SentimentAnalysis.ipynb"))
