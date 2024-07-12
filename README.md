# Data Process

## Overview

This repository deals with all things data related. Speech corpus data is preprocessed, feature encodings are extracted and saved, and data sampling and transforming is enabled.

## Scripts

Note: all alignment and VAD files are saved in seconds and all file names containing time stamps are denoted in milliseconds.

### Preprocess BuckEye

Python script name: preprocess_buckeye.py

This script extracts individual wav files for the [Buckeye](https://buckeyecorpus.osu.edu/) corpus. The data contains three splits namely: train, val (dev), and test. The splits can he found [here](https://github.com/kamperh/vqwordseg?tab=readme-ov-file#about-the-buckeye-data-splits). Alignments are found [here](https://github.com/kamperh/vqwordseg/releases/tag/v1.0). Note that the JSON files found [here](https://github.com/kamperh/zerospeech2021_baseline/tree/2f2c47766ffc02574dcc71fea7fe5247ca4f323c/datasets/buckeye) must be contained directory called beckeye_segments which is a sibling to the root Buckeye data directory.

**Example Usage**

    python3 preprocess_buckeye.py path/to/buckeye/data

### Preprocess ZRC2017

Python script name: preprocess_zrc2017.py

This script extracts individual wav files for the [ZeroSpeech](https://download.zerospeech.com/) 2017 corpus' English train split based on the VAD files found [here](). THe alignment file can be found in the test split's directory on the previously mentioned ZeroSpeech website.

**Example Usage**

    python3 preprocess_zrc2017.py path/to/zrc2017/train/data path/to/zrc2017/train/english/vad path/to/zrc2017/train/english/alignments

### Extract Feature Encodings

Python script name: encode.py

This script encodes audio by extracting its features from models (and their layers, where applicable).

**Example Usage**

    python3 wordseg/encode.py model_name path/to/audio path/to/embeddings/save --extension=.flac

The pre-trained models used are:

- wav2vec 2.0
  - [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)
  - [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)
- Hubert
  - [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
  - [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/hubert)
  - [bshall](https://github.com/bshall/hubert/tree/main)

The model_name can be one of: w2v2_fs, w2v2_hf, hubert_fs, hubert_hf, hubert_shall, melspec, mfcc. The optional extension argument is the extension of the audio files to be processed.

### Sample and Transform Data

Python script name: audio_process.py

This script contains utility functions to sample audio (and its features), to normalize sampled features, to find corresponding alignment files, and to load the alignment file attributes.