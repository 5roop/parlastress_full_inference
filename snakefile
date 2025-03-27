from pathlib import Path
hr_input_files = list(Path("data","input", "HR").glob("*.jsonl"))
rs_input_files = list(Path("data","input", "RS").glob("*.jsonl"))

hr_hashes = [i.name.split(".")[0] for i in hr_input_files]
rs_hashes = [i.name.split(".")[0] for i in rs_input_files]


audio_path_hr = "/cache/peterr/ParlaSpeeches/ParlaSpeech-HR/"
audio_path_rs = "/cache/peterr/ParlaSpeeches/ParlaSpeech-RS/"
audio_segment_path ="data/segments/"


"""
export CUDA_VISIBLE_DEVICES=4
snakemake -j 100 --use-conda --resources gpu_mem_mb=40800 --rerun-incomplete --batch gather=1/3 -k
snakemake -j 100 --use-conda --resources gpu_mem_mb=40800 --rerun-incomplete --batch gather=7/9 -k
snakemake -j 100 --use-conda --resources gpu_mem_mb=40800 --rerun-incomplete --batch gather=25/27 -k
sleep 3600
snakemake -j 100 --use-conda --resources gpu_mem_mb=40800 --rerun-incomplete -k

export CUDA_VISIBLE_DEVICES=5
snakemake -j 100 --use-conda --resources gpu_mem_mb=40800 --rerun-incomplete --batch gather=2/3 -k
snakemake -j 100 --use-conda --resources gpu_mem_mb=40800 --rerun-incomplete --batch gather=8/9 -k
snakemake -j 100 --use-conda --resources gpu_mem_mb=40800 --rerun-incomplete --batch gather=26/27 -k
"""

rule filter_jsons:
    input: "data/input/{what}/{hash}.prep.jsonl"
    output: "data/filtered/{what}/{hash}.jsonl"
    shell:
        """grep -e words_align {input[0]} > {output[0]}"""

rule prep_segments:
    input:
        jsonl=rules.filter_jsons.output,
    params:
        audio_path_hr=audio_path_hr,
        audio_path_rs=audio_path_rs,
        audio_segment_path = audio_segment_path,
    output:
        jsonl = temp("data/prep_segments/{what}/{hash}.jsonl"),
    conda: "pydub.yml"
    script: "scripts/prep_segments.py"


rule infer_on_one_json:
    input:
        jsonl=rules.prep_segments.output.jsonl,
    output:
        jsonl = "data/inferred/{what}/{hash}.jsonl",
    resources:
        gpu_mem_mb=2800,
    conda: "transformers"
    script: "scripts/infer_on_one.py"

rule post_process_inferred:
    input:
        predictions = rules.infer_on_one_json.output.jsonl,
        original = "data/input/{what}/{hash}.prep.jsonl",
    output: "data/postprocessed/{what}/{hash}.jsonl"
    conda: "transformers"
    script: "scripts/postprocess.py"

rule gather:
    default_target: True
    input: expand(rules.post_process_inferred.output,hash=hr_hashes,what=["HR" for i in hr_hashes]) + expand(rules.post_process_inferred.output,hash=rs_hashes,what=["RS" for i in rs_hashes])