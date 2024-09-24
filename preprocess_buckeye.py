"""
Extract individual wav files for Buckeye.

Author: Herman Kamper, Simon Malan
Original Repository: https://github.com/kamperh/zerospeech2021_baseline/blob/2f2c47766ffc02574dcc71fea7fe5247ca4f323c/get_buckeye_wavs.py
Contact: kamperh@gmail.com, 24227013@sun.ac.za
Date: 2021, 2024
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import json
import librosa
import soundfile as sf
import os

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "buckeye_dir", 
        type=Path, 
        help="local copy of the official Buckeye data",
        )
    return parser.parse_args()

def main():
    args = check_argv()

    in_dir = Path(args.buckeye_dir)
    out_dir = os.path.split(in_dir)[0] / Path("buckeye_segments/")

    for split in ["train","test","val"]:
        print("Extracting utterances for {} set".format(split))

        split_path = out_dir / split
        
        if not split_path.with_suffix(".json").exists():
            print("Skipping {} (no json file)".format(split))
            continue

        if split == "val": split = "dev"
        with open(split_path.with_suffix(".json")) as file:
            metadata = json.load(file)
            for in_path, start, duration, out_path in tqdm(metadata):
                wav_path = in_dir/in_path
                speaker_info = Path(out_path).stem
                speaker_info = Path(speaker_info.split("_")[0] + "_" + speaker_info.split("_")[1] + f'_{round(start*1000)}_{round((start+duration)*1000)}')

                assert wav_path.with_suffix(".wav").exists(), (
                    "'{}' does not exist".format(
                    wav_path.with_suffix(".wav"))
                    )
                
                # save alignment files IN SECONDS
                align = Path(os.path.split(in_dir)[0] / Path("buckeye_alignments/")/split/Path(out_path).stem)
                align_new = Path(os.path.split(in_dir)[0] / Path("buckeye_alignments_sec/")/split/speaker_info)
                align_new.parent.mkdir(parents=True, exist_ok=True)
                with open(align.with_suffix(".txt")) as f:
                    with open(align_new.with_suffix(".txt"), "w") as f_new:
                        for line in f:
                            start_time, end_time, word = line.strip().split(" ")
                            start_time = round(float(start_time)*0.01, 6) # take to seconds
                            end_time = round(float(end_time)*0.01, 6)
                            f_new.write(f"{start_time} {end_time} {word}\n")
                
                out_path = os.path.split(in_dir)[0] / Path("buckeye_segments/")/split/speaker_info
                out_path.parent.mkdir(parents=True, exist_ok=True)
                wav, _ = librosa.load(
                    wav_path.with_suffix(".wav"), sr=16000,
                    offset=start, duration=duration
                    )
                sf.write(
                    out_path.with_suffix(".wav"), wav, samplerate=16000)

if __name__ == "__main__":
    main()