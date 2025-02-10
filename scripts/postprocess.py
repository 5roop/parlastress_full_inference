try:
    predictions = snakemake.input.predictions
    original = snakemake.input.original
    output = snakemake.output[0]
except NameError as e:
    predictions = "data/inferred/A44UFMcUAMU.jsonl"
    original = "data/input/A44UFMcUAMU.prep.jsonl"
    output = "data/postprocessed/A44UFMcUAMU.jsonl"

import polars as pl
import numpy as np

pl.Config.set_tbl_cols(-1)
<<<<<<< HEAD
pl.Config.set_tbl_width_chars(600)
=======
pl.Config.set_tbl_width_chars(700)
>>>>>>> 067fecb8ce717d1352404dbc8c58e573a68c088a
pl.Config.set_fmt_str_lengths(50)


original = pl.read_ndjson(original, infer_schema_length=None)
predictions = pl.read_ndjson(predictions).with_columns(
    pl.col("segment_files")
    .str.split("_")
    .list.last()
    .str.split(".")
    .list.first()
    .cast(pl.Int32)
    .alias("word_idx")
)

assert predictions.filter(pl.col("primary_stress").list.len() == 0).shape[0] == 0, (
    "There are predictions with no stress found!"
)


def backpropagate_predictions(row):
    rowid = row["id"]
    subset = predictions.filter(pl.col("id").eq(rowid))
    # Edge case: if there is nothing to align, don't align
    if subset.shape[0] == 0:
        return None
    return_list = []
    for subsetrow in subset.iter_rows(named=True):
        idx = subsetrow["word_idx"]
        start, stop = subsetrow["primary_stress"]
        multisyllabic_word = [
            i for i in row["multisyllabic_words"] if i["word_idx"] == idx
        ][0]
        nuclei = multisyllabic_word["nuclei"]
        charalign = row["chars_align"][idx]
        word = row["words_align"][idx]
        offset = float(word["time_s"])
        start, stop = start + offset, stop + offset
        target = (start + stop) / 2
        candidates = [charalign[i] for i in nuclei]
        candidates_centroids = [(i["time_s"] + i["time_e"]) / 2 for i in candidates]
        differences = [(i - target) ** 2 for i in candidates_centroids]
        winner_index = np.argmin(differences)
        winner = candidates[winner_index]
        winner_index = charalign.index(winner)
        return_list.append({"word_idx": idx, "stress": winner_index, "nuclei": nuclei})
    return return_list
    2 + 2


# a = list(original.iter_rows(named=True))[0]
# backpropagate_predictions(a)

return_dtype = pl.List(
    pl.Struct(
        [
            pl.Field("word_idx", pl.UInt16),
            pl.Field("stress", pl.UInt16),
            pl.Field("nuclei", pl.List(pl.UInt16)),
        ]
    )
)

original = original.with_columns(
    pl.struct(["id", "chars_align", "words_align", "multisyllabic_words"])
    .map_elements(backpropagate_predictions, return_dtype=return_dtype)
    .alias("multisyllabic_words")
)
2 + 2

original.write_ndjson(
    output,
)
