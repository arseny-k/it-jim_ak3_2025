# 🎧 Task 3 - Audio-Based Music Playlist Clustering

This project clusters songs into playlists based on their **audio similarity**, using both classical audio features and deep learning embeddings.

Ever thrown all your songs into one giant playlist? What if a tool could organize them automatically by how they sound? That’s what this project does.

---

## 📁 Project Structure

```
.
├── main_v1.py                         # Clustering using MFCC + tempo
├── main_v2.py                         # Clustering using CNN14 deep embeddings
├── models/
│   ├── Cnn14_mAP=0.431.pth            # Pretrained CNN14 model weights
│   └── audioset_tagging_cnn/          # Cloned model definition repo (PANNs)
├── playlists_v1.json                  # Output from Version 1 (example)
├── playlists_v2.json                  # Output from Version 2 (example)
├── requirements.txt                   # Python dependencies
└── report.pdf                         # Project report explaining methodology
```

---

## ⚙️ Setup Instructions

0. **Clone the repo and navigate to the project root:**

```bash
git clone https://github.com/arseny-k/it-jim_ak3_2025
cd it-jim_ak3_2025
```

1. **Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up the CNN14 model (for main_v2.py):**

```bash
# Clone the model repo
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git models/audioset_tagging_cnn

# Download CNN14 weights
wget -O models/Cnn14_mAP=0.431.pth https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
```

---

## 🚀 How to Use

### 🎛️ Common Options

- `--path`: Path to folder containing your songs
- `--n`: Number of playlists (clusters) to generate
- `--output`: Where to save the resulting JSON file
- `--silent`: Suppress all logs except warnings/errors
- `--logfile`: Optional path to log output to a file
- `--debug`: Show extra debug info (shapes, progress, etc.)

---

### 🔉 Version 1 — MFCC + Tempo

```bash
python main_v1.py --path data/songs --n 3 --output playlists_v1.json
```

---

### 🧠 Version 2 — Deep Embeddings (CNN14)

```bash
python main_v2.py --path data/songs --n 3 --output playlists_v2.json
```

Optional:
```bash
# With logging and model path
python main_v2.py \
  --path data/songs \
  --n 3 \
  --checkpoint models/Cnn14_mAP=0.431.pth \
  --output playlists_v2.json \
  --debug \
  --logfile logs/run.log
```

---

## 📦 Output Format

Each script will generate a JSON file like this:

```json
{
  "playlists": [
    {
      "id": 0,
      "songs": ["song1.mp3", "song2.mp3"]
    },
    {
      "id": 1,
      "songs": ["song3.mp3"]
    }
  ]
}
```

---

## 🔧 Requirements

Python ≥ 3.8  
See `requirements.txt` for full dependency list.

---

## 🧹 Notes

- Supports most common audio formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`
- Output playlists are based on how the music sounds, not metadata

---

## 🧠 Credit

- Deep model from [PANNs (CNN14)](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- Developed as part of an audio similarity and machine learning task

---

## 🚀 Author
Made with ❤️ by Arseny K
