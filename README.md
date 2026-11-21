# MELD sentiment and emotion analysis

- [Submission notebook](MELD_Submission.ipynb) contains the pipeline and saved outputs. Download it and use **Run All** in its folder to recalculate the evaluation from the embedded inputs. Optional model retraining needs internet access and a CUDA GPU.

On the 2,610-utterance MELD test set, adding audio to Whisper text raised sentiment macro F1 from 58.65% to 60.90% and emotion weighted F1 from 54.38% to 55.70%.

The notebook includes transcripts and labels from the [MELD dataset](https://huggingface.co/datasets/declare-lab/MELD) via the [MELD audio mirror](https://huggingface.co/datasets/ajyy/MELD_audio), both marked GPL-3.0. The upstream terms are in [MELD_DATA_LICENSE.txt](MELD_DATA_LICENSE.txt). Cite [Poria et al. (2019)](https://aclanthology.org/P19-1050/) when using MELD.
