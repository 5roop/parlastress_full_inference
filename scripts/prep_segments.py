try:
    injson = snakemake.input.jsonl
    output = snakemake.output.jsonl
    audio_path_hr = snakemake.params.audio_path_hr
    audio_path_rs = snakemake.params.audio_path_rs
    audio_segment_path = snakemake.params.audio_segment_path
except NameError:
    injson = "data/filtered/HR/_fxXNGLg-4U.jsonl"
    output = "data/prep_segments/_fxXNGLg-4U.jsonl"
    audio_path_hr = "/cache/peterr/ParlaSpeeches/ParlaSpeech-HR/"
    audio_path_rs = "/cache/peterr/ParlaSpeeches/ParlaSpeech-RS/"
    audio_segment_path = "data/segments/"

import polars as pl
from pathlib import Path
from pydub import AudioSegment

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(900)
df = pl.read_ndjson(injson)
if df["id"].str.starts_with("ParlaMint-HR").all():
    df = df.with_columns((audio_path_hr + "/" + pl.col("audio")).alias("audio"))
elif df["id"].str.starts_with("ParlaMint-RS").all():
    df = df.with_columns((audio_path_rs + "/" + pl.col("audio")).alias("audio"))
else:
    raise AttributeError("IDs do not seem to start with either RS or HR!")


assert (
    df["audio"].map_elements(lambda s: Path(s).exists(), return_dtype=pl.Boolean).all()
), "Missing flacs!"


def get_indices(row) -> list[int]:
    words = row["words_align"]
    multisyllabic = row["multisyllabic_words"]
    indices = [i["word_idx"] for i in multisyllabic]
    return indices


def get_segment_names(row) -> list[str]:
    results = []
    indices = get_indices(row)
    audio_basename = Path(row["audio"]).with_suffix("").name
    audio = AudioSegment.from_file(row["audio"])
    for i in indices:
        path = Path(audio_segment_path, audio_basename + f"_{i}.wav")
        # if path.exists():
        if False:
            pass
        else:
            w = row["words_align"][i]
            word_start_ms = int(1000 * w["time_s"])
            word_end_ms = int(1000 * w["time_e"])
            path.parent.mkdir(exist_ok=True)
            audio[word_start_ms:word_end_ms].export(path, format="wav")
        results.append(str(path))
    return results


# df = df.filter(pl.col("id").eq("ParlaMint-HR_2016-02-03-0.u1815_1111-1302"))
# for row in df.iter_rows(named=True):
#     get_segment_names(row)

df = df.with_columns(
    pl.struct(["words_align", "multisyllabic_words", "audio"])
    .map_elements(get_segment_names, return_dtype=pl.List(pl.String))
    .alias("segment_files")
)
2 + 2
df.write_ndjson(output)
