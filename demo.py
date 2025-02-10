import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
import torch
import torchaudio

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name = "5roop/Wav2Vec2BertPrimaryStressAudioFrameClassifier"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(model_name).to(device)

wavfile = "data/segments/A44UFMcUAMU_328.3-332.8_0.wav"
waveform, sampling_rate = torchaudio.load(wavfile)
print(sampling_rate)

with torch.no_grad():
    inputs = feature_extractor(
        waveform, return_tensors="pt", sampling_rate=sampling_rate
    ).to(device)
    logits = model(**inputs).logits.cpu()
frames = np.argmax(logits, axis=-1).reshape(-1)


def frames_to_intervals(frames: list[int]) -> list[tuple[float]]:
    from itertools import pairwise
    import pandas as pd

    results = []
    ndf = pd.DataFrame(
        data={
            "time_s": [0.020 * i for i in range(len(frames))],
            "frames": frames,
        }
    )
    ndf = ndf.dropna()
    indices_of_change = ndf.frames.diff()[ndf.frames.diff() != 0].index.values
    for si, ei in pairwise(indices_of_change):
        if ndf.loc[si : ei - 1, "frames"].mode()[0] == 0:
            pass
        else:
            results.append(
                (round(ndf.loc[si, "time_s"], 3), round(ndf.loc[ei - 1, "time_s"], 3))
            )
    if results == []:
        return None
    # Post-processing: if multiple regions were returned, only the longest should be taken:
    if len(results) > 1:
        results = sorted(results, key=lambda t: t[1] - t[0], reverse=True)
    return results[0]


intervals = frames_to_intervals(frames)

print(intervals)


# A better way (in many ways): datasets:

from datasets import Audio, Dataset


def evaluator(chunks):
    sampling_rate = chunks["audio"][0]["sampling_rate"]
    with torch.no_grad():
        inputs = feature_extractor(
            [i["array"] for i in chunks["audio"]],
            return_tensors="pt",
            sampling_rate=sampling_rate,
        ).to(device)
        logits = model(**inputs).logits
    y_pred_raw = np.array(logits.cpu())
    y_pred = y_pred_raw.argmax(axis=-1)
    primary_stress = [frames_to_intervals(i) for i in y_pred]
    return {
        "y_pred": y_pred,
        "y_pred_logits": y_pred_raw,
        "primary_stress": primary_stress,
    }


# Create a dataset and map our evaluator function on it:
ds = Dataset.from_dict(
    {
        "audio": [
            "data/segments/A44UFMcUAMU_328.3-332.8_0.wav",
            "data/segments/A44UFMcUAMU_328.3-332.8_1.wav",
        ]
    }
).cast_column("audio", Audio(16000, mono=True))
ds = ds.map(
    evaluator,
    batched=True,  # This means evaluator function will be given a list of instances
    batch_size=1,
    remove_columns=["audio"]
)  # Adjust batch size according to your hardware specs
print(ds["y_pred"][0])
# Outputs: [0, 0, 1, 1, 1, 1, 1, ...]
print(ds["y_pred_logits"][0])
# Outputs:
# [[ 0.89419061, -0.77746612],
#  [ 0.44213724, -0.34862748],
#  [-0.08605709,  0.13012762],
# ....
print(ds["primary_stress"][0])
# Outputs: [0.34, 0.4]

2+2
