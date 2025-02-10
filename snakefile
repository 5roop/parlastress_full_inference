from pathlib import Path
input_files = list(Path("data","input").glob("*.jsonl"))
hashes = [i.name.split(".")[0] for i in input_files]

audio_path = "/cache/nikolal/parlaspeech-hr/repository"
audio_segment_path ="data/segments/"
"""
export CUDA_VISIBLE_DEVICES=6;
snakemake -j 100 --use-conda --resources jobs_per_gpu=5

"""

configfile: "snakeconfig.yml"

rule filter_jsons:
    input: "data/input/{hash}.prep.jsonl"
    output: "data/filtered/{hash}.jsonl"
    shell:
        """grep -e words_align {input[0]} > {output[0]}"""

rule prep_segments:
    input:
        jsonl=rules.filter_jsons.output
    params:
        audio_path=audio_path,
        audio_segment_path = audio_segment_path,
    output:
        jsonl = "data/prep_segments/{hash}.jsonl",
    conda: "transformers"
    script: "scripts/prep_segments.py"


rule infer_on_one_json:
    input:
        jsonl=rules.prep_segments.output.jsonl,
    params:
        audio_path=audio_path,
    output:
        jsonl = "data/inferred/{hash}.jsonl",
    resources:
        jobs_per_gpu=1,
    conda: "transformers"
    script: "scripts/infer_on_one.py"

rule post_process_inferred:
    input:
        predictions = rules.infer_on_one_json.output.jsonl,
        original = "data/input/{hash}.prep.jsonl",
    output: "data/postprocessed/{hash}.jsonl"
    conda: "transformers"
    script: "scripts/postprocess.py"

rule gather:
    default_target: True
    input: expand(rules.post_process_inferred.output, hash=hashes)