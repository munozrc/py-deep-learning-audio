"""
Create Audio Dataset for DeepLearning
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
"""
import json
import os
import math
import librosa

DATASET_PATH = "audios"
JSON_PATH = "music-genre-dataset.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


# dictionary to store mapping, labels, and MFCCs
data = {
    "mapping": [],
    "labels": [],
    "mfcc": []
}

NUM_SEGMENTS = 10
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 14

SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
num_mfcc_vectors_per_segment = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)

# loop through all genre sub-folder
for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):

    # ensure we're processing a genre sub-folder level
    if dirpath is DATASET_PATH:
        continue

    # save genre label (i.e., sub-folder name) in the mapping
    semantic_label = dirpath.split("\\")[-1]
    data["mapping"].append(semantic_label)
    print(f"\nProcessing: {semantic_label}")

    # process all audio files in genre sub-dir
    for f in filenames:
        file_path = os.path.join(dirpath, f)
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        # process all segments of audio file
        for d in range(NUM_SEGMENTS):

            # calculate start and finish sample for current segment
            start = SAMPLES_PER_SEGMENT * d
            finish = start + SAMPLES_PER_SEGMENT

            # extract mfcc
            mfcc = librosa.feature.mfcc(
                y=signal[start:finish], sr=sample_rate, n_mfcc=N_MFCC,
                n_fft=N_FFT, hop_length=HOP_LENGTH)
            mfcc = mfcc.T

            # store only mfcc feature with expected number of vectors
            if len(mfcc) == num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(i-1)
                print(f"{file_path}, segment:{d+1}")

# save MFCCs to json file
with open(JSON_PATH, "w", encoding="utf-8") as fp:
    json.dump(data, fp, indent=4)
