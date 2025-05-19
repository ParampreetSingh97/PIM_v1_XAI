# Prasar Bharati Indian Music Version-1 (PIM-v1) Dataset & Code for Raga Identification and Explainable AI analysis

This repository contains the codes and metadata for our paper:
**"[Explainable Deep Learning Analysis for Raga Identification in Indian Art Music]"**  
📄 [https://doi.org/10.48550/arXiv.2406.02443]
   [P. Singh and V. Arora, “Explainable deep learning analysis for raga identification in indian art music,” arXiv preprint arXiv:2406.02443, 2024.]

Full dataset is available at: **https://zenodo.org/records/15461509**

## 📁 Dataset Summary
- **501** Hindustani Classical Music concert recordings, split into 30 second clips
- Provided as **log-magnitude spectrograms**
- Sampling Rate: 16 kHz
- Spectrogram Params:
  - Window Length = 80 ms
  - Hop Length = 40 ms
  - Max Frequency = 5000 Hz
  - Clip Duration = 30 sec

## 🧾 Metadata Fields
`Audio_file`, `Raga Name`, `Tonic`, `Artist Hindustani Music`, `Gharana`, `Tala`, `Vocals`, `Composer`, `Participants`, `Occasion`, ...

See full list in [`metadata.csv`](metadata.csv).

## 🔧 Usage

```python
from dataset_utils.load_dataset import load_metadata
df = load_metadata('metadata.csv')

