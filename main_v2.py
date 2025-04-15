# main_v2.py - Deep Audio Clustering using CNN14 embeddings

import os
import argparse
import json
import logging
import sys

import torch
import torchaudio
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add PANNs model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'audioset_tagging_cnn', 'pytorch'))
from models import Cnn14


def setup_logger(silent=False, debug=False, logfile=None):
    handlers = []
    if not silent:
        handlers.append(logging.StreamHandler())
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode='w'))

    level = logging.DEBUG if debug else (logging.WARNING if silent else logging.INFO)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=level, handlers=handlers)


def load_model(checkpoint_path, device):
    logging.info("Loading CNN14 model...")
    model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully.")
    return model


def extract_embedding(model, waveform, device):
    with torch.no_grad():
        waveform = waveform.to(device)
        output_dict = model(waveform)
        embedding = output_dict['embedding']
        return embedding.squeeze(0).cpu().numpy()


def cluster_embeddings(embeddings, song_ids, n_clusters):
    logging.info("Clustering with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    playlists = [{"id": i, "songs": []} for i in range(n_clusters)]
    seen = set()
    for song_id, label in zip(song_ids, labels):
        if song_id not in seen:
            playlists[label]["songs"].append(song_id)
            seen.add(song_id)
    return {"playlists": playlists}


def main():
    parser = argparse.ArgumentParser(description="Cluster songs using CNN14 deep audio embeddings")
    parser.add_argument("--path", type=str, required=True, help="Path to folder with audio files")
    parser.add_argument("--n", type=int, required=True, help="Number of playlists (clusters)")
    parser.add_argument("--checkpoint", type=str, default="models/Cnn14_mAP=0.431.pth", help="Path to CNN14 model weights")
    parser.add_argument("--output", default="playlists_v2.json", help="Output JSON path")
    parser.add_argument("--silent", action="store_true", help="Suppress console output")
    parser.add_argument("--logfile", type=str, help="Optional path to save logs")
    parser.add_argument("--debug", action="store_true", help="Enable debug-level logging")
    args = parser.parse_args()

    setup_logger(silent=args.silent, debug=args.debug, logfile=args.logfile)

    logging.info("Starting Version 2 clustering (Deep audio features)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    file_list = [f for f in os.listdir(args.path) if f.lower().endswith(AUDIO_EXTENSIONS)]

    if not file_list:
        logging.error("No audio files found in the specified directory.")
        return

    embeddings = []
    song_ids = []

    for filename in tqdm(file_list, desc="Extracting embeddings"):
        filepath = os.path.join(args.path, filename)
        try:
            logging.debug(f"Loading {filename}")
            waveform, sr = torchaudio.load(filepath)

            if sr != 32000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            embedding = extract_embedding(model, waveform, device)
            logging.debug(f"Embedding shape: {embedding.shape}")
            embeddings.append(embedding)
            song_ids.append(filename)

        except Exception as e:
            logging.warning(f"Skipping {filename}: {e}")

    embeddings = np.vstack(embeddings)
    logging.info(f"Feature matrix shape: {embeddings.shape}")

    result = cluster_embeddings(embeddings, song_ids, args.n)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)

    logging.info("Playlists saved to %s", args.output)
    logging.info("Process finished successfully.")


if __name__ == "__main__":
    main()
