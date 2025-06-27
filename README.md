# Prasar Bharati Indian Music Version-1 (PIM-v1) Dataset & Code for Raga Identification and Explainable AI analysis

This repository contains the codes and metadata for our paper:
**"[Explainable Deep Learning Analysis for Raga Identification in Indian Art Music]"**  
📄 [https://doi.org/10.1109/TASLPRO.2025.3574839]
   [P. Singh and V. Arora, "Explainable Deep Learning Analysis for Raga Identification in Indian Art Music," in IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 2302-2311, 2025]

Full dataset will be available soon

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

