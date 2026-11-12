# Sentiment and emotion in MELD conversations

This NLP course project compares human and Whisper transcripts on the same MELD utterances. RoBERTa reads the target and nearby dialogue turns. A logistic regression classifier uses emotion2vec+ audio features. The project also extracts representative quotes and lists each speaker's predicted emotion changes.

## Final report

- [MELD_Final.ipynb](MELD_Final.ipynb) contains the executed evaluation, class breakdown, and summary examples.
- [final_results.csv](outputs/final_results.csv) gives the full metric table. [final_settings.json](outputs/final_settings.json) records the models and fusion weights selected on development data.

The matched transcript comparison holds the checkpoint fixed. The context comparison holds the base model and training schedule fixed. Training uses 9,988 utterances with audio, development uses 1,108, and testing uses all 2,610.

## Run locally

The recorded run used Python 3.12 and an NVIDIA GPU. These Windows commands install the same CUDA build of PyTorch.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m ipykernel install --user --name sentiment-nlp
```

Open `MELD_Final.ipynb` at the project root. Its saved results can be viewed directly. Rerunning uses the MELD CSVs, local Whisper transcript caches, and saved prediction CSVs.

For a fresh run, execute `SentimentAnalysis.ipynb` first. It downloads MELD, prepares local Whisper turbo transcripts, and computes the baseline predictions. It runs locally after the data and models download. To retrain the text models, run the following commands.

```powershell
.venv/Scripts/python.exe meld_context.py --mode gold
.venv/Scripts/python.exe meld_context.py --mode mixed
.venv/Scripts/python.exe meld_context.py --mode mixed --current-only
.venv/Scripts/python.exe meld_context.py --mode mixed --model-name FacebookAI/roberta-large --epochs 3 --batch-size 8 --lr 0.00001
```

The recorded RoBERTa-large run used a CUDA GPU with BF16 support. The development table also includes the GoEmotions initialization trial when its predictions are available (`--mode mixed --model-name SamLowe/roberta-base-go_emotions`).

Run `MELD_Final.ipynb` again to score the saved predictions.

`build_notebook.py` generates the preparation notebook. Git excludes the dataset, transcript caches, and model weights. A teammate can read the saved final notebook and result tables immediately.

[MELD paper](https://aclanthology.org/P19-1050/) · [MELD audio files](https://huggingface.co/datasets/ajyy/MELD_audio)
