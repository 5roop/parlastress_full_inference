try:
    injson = snakemake.input.jsonl
    output = snakemake.output.jsonl
except NameError:
    injson = "data/prep_segments/HR/goZfMp5DWCg.jsonl"
    output = "brisi.jsonl"

import polars as pl
from pathlib import Path
from tqdm import tqdm

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(900)
pl.Config.set_fmt_str_lengths(100)
df = pl.read_ndjson(injson)


import numpy as np

from datasets import Audio, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Inferring on cuda")
else:
    device = torch.device("cpu")

model_name = "5roop/Wav2Vec2BertPrimaryStressAudioFrameClassifier"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(model_name).to(device)


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
                (round(ndf.loc[si, "time_s"], 3), round(ndf.loc[ei, "time_s"], 3))
            )
    if results == []:
        return results
    # Post-processing: if multiple regions were returned, only the longest should be taken:
    if len(results) > 1:
        results = sorted(results, key=lambda t: t[1] - t[0], reverse=True)
    return results[0]


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


df = (
    df.select(["segment_files", "id"])
    .filter(pl.col("segment_files").list.len() > 0)
    .explode("segment_files")
    .with_columns(pl.col("segment_files").alias("audio"))
)


ds = Dataset.from_pandas(df.to_pandas()).cast_column("audio", Audio(16000, mono=True))
for i in ds:
    if i["audio"] is None:
        print(i)
from tqdm import tqdm

try:
    ds = ds.map(evaluator, batched=True, batch_size=1)
    ds.to_polars().select(["segment_files", "id", "primary_stress"]).write_ndjson(
        output
    )
except TypeError:
    ps = []
    for i in tqdm(range(len(ds)), total=len(ds)):
        ps.append(evaluator(ds[i : i + 1])["primary_stress"])
    ds.to_polars().with_columns(
        primary_stress=ps,
    ).select(["segment_files", "id", "primary_stress"]).write_ndjson(output)


for i in tqdm(df["audio"].to_list(), desc="Removing segments"):
    Path(i).unlink()
2 + 2
