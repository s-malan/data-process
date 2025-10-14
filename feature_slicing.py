"""
Slice speech features and corresponding TextGrids into sub-utterances based on silences in the alignments.
Acts as a strict VAD system.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: October 2025
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob
from tqdm import tqdm

import numpy as np
import torch
import textgrids
from syllabify import syllabify

def load_features(file):
    """
    Load the specified speech feature from file path

    Parameters
    ----------
    file : String
        File path to the speech feature

    Return
    ------
    feature : Tensor
        Speech features in a tensor (num_frames, num_features)
    """

    feature = torch.from_numpy(np.load(file))
    if len(feature.shape) == 1: # if only one dimension, add a dimension
        feature = feature.unsqueeze(0)
    return feature

def output_segment(extact_feat, extact_grid, features_dir, align_file, align_out_path, words, current_grid, num_sub_utterances):
    """
    Extract and save the current sub-utterance's features and TextGrid.

    Parameters
    ----------
    extact_feat : bool
        Whether to extract features for each sub-utterance
    extact_grid : bool
        Whether to extract TextGrids for each sub-utterance
    features_dir : Path
        Directory containing speech features
    align_file : String
        File path to the alignment TextGrid
    align_out_path : Path
        Output path for the sub-utterance TextGrid
    words : List (Interval)
        List of word intervals in the current sub-utterance
    current_grid : TextGrid
        Current TextGrid being processed
    num_sub_utterances : int
        Number of sub-utterances extracted so far from the current utterance
    """

    current_grid.xmin = words[0].xmin
    current_grid.xmax = words[-1].xmax

    if extact_feat:
        # Slice features and save
        feature_file = glob(os.path.join(features_dir, f'**/{Path(align_file).stem}.npy'), recursive=True)[0]
        feature = load_features(feature_file)
        feature_xmin = int(np.round(current_grid.xmin / 20 * 1000))
        feature_xmax = int(np.round(current_grid.xmax / 20 * 1000))
        feature = feature[feature_xmin:feature_xmax, :]
        feature_file = os.path.relpath(feature_file, features_dir)
        features_dir_out = "/".join(str(features_dir).split('/')[:-2]) + "_feature_sliced/" + "/".join(str(features_dir).split('/')[-2:])
        features_dir_out = Path(features_dir_out) / Path(feature_file.split('.')[0] + f'-{num_sub_utterances:04d}.npy')
        feature_file = Path(str(features_dir_out / Path(align_file).stem) + f'-{num_sub_utterances:04d}.npy')
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_file, feature.cpu().numpy())

    if extact_grid:
        # New TextGrid
        current_grid["words"] = textgrids.Tier(words)

        # Get corresponding phones and syllables
        phones = [p for p in textgrids.TextGrid(align_file)["phones"] if p.xmin >= current_grid.xmin and p.xmax <= current_grid.xmax]
        current_grid["phones"] = textgrids.Tier(phones)
        # Syllabify
        phone_transcription = [p.text for p in phones]
        phone_xmins = [p.xmin for p in phones]
        phone_xmaxs = [p.xmax for p in phones]
        syllables = syllabify.syllabify(phone_transcription)
        syl_intervals = textgrids.Tier()
        for syl in syllables:
            syl = [item for sublist in syl for item in sublist]
            syl_len = len(syl)
            syl_intervals.append(textgrids.Interval(' '.join(syl), phone_xmins[0], phone_xmaxs[syl_len-1]))
            del phone_xmins[:syl_len]
            del phone_xmaxs[:syl_len]
        current_grid["syllables"] = syl_intervals

        # Save sub-utterance TextGrid
        align_out_path_new = Path(str(align_out_path) + f'-{num_sub_utterances:04d}.TextGrid')
        align_out_path_new.parent.mkdir(parents=True, exist_ok=True)
        current_grid.write(align_out_path_new)

def segment_data(args):
    """
    Loops through all alignment files, splits them and/or the corresponding features into sub-utterances based on silences in the alignments.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    extact_feat = args.extact_feat
    extact_grid = args.extact_grid
    features_dir = args.feature_dir
    align_dir = args.alignments_dir
    align_out_dir = align_dir.parent / Path(str(align_dir).split('/')[-1] + "_feature_sliced")

    # Split each utterance into sub-utterances based on silences in the alignments
    for align_file in tqdm(sorted(glob(os.path.join(align_dir, f'**/*.TextGrid'), recursive=True))):
        current_grid = textgrids.TextGrid(align_file)
        align_out_path = align_out_dir / Path(align_file).stem
        words = []
        num_sub_utterances = 0
        for word in textgrids.TextGrid(align_file)["words"]:
            if word.text in ["<unk>", ""]: # Save previous sub-utterance and start new one
                if len(words) == 0: continue
                output_segment(extact_feat, extact_grid, features_dir, align_file, align_out_path, words, current_grid, num_sub_utterances)

                words = []
                current_grid = textgrids.TextGrid(align_file)
                num_sub_utterances += 1
            else:
                # Build sub-utterance
                words.append(word)
                
        # Save last sub-utterance (if present)
        if len(words) > 0:
            output_segment(extact_feat, extact_grid, features_dir, align_file, align_out_path, words, current_grid, num_sub_utterances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "feature_dir",
        type=Path,
        help="speech features directory"
        )
    parser.add_argument(
        "alignments_dir",
        type=Path,
        help="alignment files corresponding to features"
        )
    parser.add_argument(
        "extact_feat",
        type=bool,
        help="whether to extract TextGrids for each sub-utterance"
        )
    parser.add_argument(
        "extact_grid",
        type=bool,
        help="whether to extract TextGrids for each sub-utterance"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()

    segment_data(args)