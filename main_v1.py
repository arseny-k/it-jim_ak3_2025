# main_v1.py - Classical Audio Clustering (MFCC + Tempo)

import os
import argparse
import json
import logging
from pathlib import Path

import librosa
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def setup_logger(silent=False, debug=False, logfile=None):
    handlers = []
    if not silent:
        handlers.append(logging.StreamHandler())
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode='w'))

    level = logging.DEBUG if debug else (logging.WARNING if silent else logging.INFO)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=level, handlers=handlers)


def extract_features(file_path, sr=22050, n_mfcc=13):
    """
    Extract MFCC features and tempo from an audio file.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        if y.size == 0:
            raise ValueError("Empty audio signal")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feature_vector = np.hstack([np.mean(mfcc, axis=1), tempo])
        return feature_vector
    except Exception as e:
        raise RuntimeError(f"Error extracting features from {file_path}: {e}")


def cluster_songs(folder_path, n_clusters):
    """
    Extract features from audio files and cluster them.
    """
    AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    song_paths = [p for ext in AUDIO_EXTENSIONS for p in Path(folder_path).rglob(f'*{ext}')]

    if not song_paths:
        logging.error("No audio files found in the specified directory.")
        return {"playlists": []}

    features = []
    filenames = []

    logging.info("Extracting features from %d songs...", len(song_paths))
    for song_path in tqdm(song_paths, desc="Extracting features"):
        try:
            logging.debug(f"Processing {song_path.name}")
            feats = extract_features(song_path)
            features.append(feats)
            filenames.append(song_path.name)
        except Exception as e:
            logging.warning(f"Skipping {song_path.name}: {e}")

    features = np.array(features)
    logging.info(f"Feature matrix shape: {features.shape}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    logging.info("Clustering with KMeans...")
    labels = kmeans.fit_predict(features)

    playlists = [{"id": i, "songs": []} for i in range(n_clusters)]
    seen = set()
    for fname, label in zip(filenames, labels):
        if fname not in seen:
            playlists[label]["songs"].append(fname)
            seen.add(fname)

    return {"playlists": playlists}


def save_playlists(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info("Playlists saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Cluster songs using MFCC + Tempo features")
    parser.add_argument("--path", required=True, help="Path to folder with audio files")
    parser.add_argument("--n", type=int, required=True, help="Number of clusters")
    parser.add_argument("--output", default="playlists_v1.json", help="Output JSON path")
    parser.add_argument("--silent", action="store_true", help="Suppress console output")
    parser.add_argument("--logfile", type=str, help="Optional path to save logs")
    parser.add_argument("--debug", action="store_true", help="Enable debug-level logging")
    args = parser.parse_args()

    setup_logger(silent=args.silent, debug=args.debug, logfile=args.logfile)

    logging.info("Starting Version 1 clustering (Classical audio features)...")
    result = cluster_songs(args.path, args.n)
    save_playlists(result, args.output)
    logging.info("Process finished successfully.")


if __name__ == "__main__":
    main()
