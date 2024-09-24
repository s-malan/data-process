"""
Extract individual wav files for ZRC2017 data using VAD. Extract individual alignment files for ZRC2017.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: May 2024
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import soundfile as sf
import sys
import os

def segment_data(args):
    in_dir = args.zrc_dir / Path(args.language)
    out_dir = os.path.split(args.zrc_dir)[0]
    out_dir = out_dir / Path("zrc2017_train_segments/") / Path(args.language)
    align_out_dir = os.path.split(os.path.split(args.zrc_dir)[0])[0] / Path("zrc_alignments/zrc2017_train_alignments/") / Path(args.language) 

    with open(args.vad_dir) as file:
        for index, line in tqdm(enumerate(file)):
            if index == 0: # skip header
                continue

            file_name, start_seconds, end_seconds = line.strip().split(",")
            start_seconds = float(start_seconds)
            end_seconds = float(end_seconds)
            dur = end_seconds - start_seconds
            wav_path = in_dir / Path(file_name).with_suffix(".wav")
            out_details = Path(file_name).stem + "_" + str(round(start_seconds*1000)) + "-" + str(round(end_seconds*1000))
            align_out_path = (align_out_dir / out_details).with_suffix(".txt")

            if start_seconds == end_seconds: # the file is empty
                continue

            if wav_path.exists(): # and not out_path.with_suffix(".wav").exists(): # in train set and not already extracted
                out_path = out_dir / out_details
                out_path.parent.mkdir(parents=True, exist_ok=True)
                wav, _ = librosa.load(
                    wav_path, sr=16000, offset=start_seconds, duration=dur
                    )
                sf.write(out_path.with_suffix(".wav"), wav, samplerate=16000)
            
            # Alignments:
            # if not os.path.isfile(align_out_path): # split alignment data
            align_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(align_out_path, "w") as align_out_file:
                with open(args.alignments_dir) as alignment_file:
                    curr_start = None
                    for alignment_line in alignment_file:
                        alignment_file_name, alignment_start_seconds, alignment_end_seconds, word = alignment_line.strip().split(" ")
                        if alignment_file_name == file_name and round(float(alignment_start_seconds), 2) >= start_seconds and round(float(alignment_end_seconds), 2) <= end_seconds:
                            if curr_start is None:
                                curr_start = start_seconds
                                vad_align_discrepency = abs(float(alignment_start_seconds) - curr_start)
                                relative_start = round((float(alignment_start_seconds) - curr_start - vad_align_discrepency), 6) # save in s
                            else:
                                relative_start = round((float(alignment_start_seconds) - curr_start), 6) # save in s
                            relative_end = round((float(alignment_end_seconds) - curr_start), 6) # save in s
                            align_out_file.write(f"{relative_start} {relative_end} {word}\n")
                        elif curr_start is not None: # end of segment
                            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "zrc_dir",
        type=Path,
        help="local copy of the official ZRC2017 data"
        )
    parser.add_argument(
        "vad_dir",
        type=Path,
        help="VAD files for ZRC2017 data"
        )
    parser.add_argument(
        "alignments_dir",
        type=Path,
        help="alignment file for ZRC2017 data"
        )
    parser.add_argument(
        "language",
        options=["english", "french", "mandarin", "german", "wolof"],
        type=str,
        help="ZRC2017 language",
        default="english"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()

    segment_data(args)