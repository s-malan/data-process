# Data Process

This repository is responsible for the preprocessing of speech data. This enables lightweight audio encodings involved in downstream tasks.

## Scripts

Note: all alignment and VAD files are saved in seconds and all file names containing time stamps are denoted in milliseconds.

### Feature Slicing

Python script name: `feature_slicing.py`

This script slices speech features and alignment files (mainly used for the LibriSpeech corpus [https://www.openslr.org/12](https://www.openslr.org/12)) based on the silences found in the alignments ([https://zenodo.org/records/2619474](https://zenodo.org/records/2619474)). Syllable alignments are created and added to the TextGrid files (requires the repo at [https://github.com/kylebgorman/syllabify](https://github.com/kylebgorman/syllabify)).

**Example Usage**

    python3 feature_slicing.py path/to/alignment/data slice_features --features_dir=path/to/speech/features --extract_grid
  
This script assumes a similar path structure for features `/.../.../librispeech/dev_clean/model/layer`, and for alignments `/.../.../librispeech_alignments/dev_clean`. `extract_grid` (default: `True`) chooses if alignments should be split, the presence of `features_dir` decides if features are split or not.

### Preprocess BuckEye

Python script name: `preprocess_buckeye.py`

This script extracts individual wav files for the BuckEye [https://buckeyecorpus.osu.edu/](https://buckeyecorpus.osu.edu/) corpus. The data contains three splits namely: train, val (dev), and test. The splits can he found here [https://github.com/kamperh/vqwordseg?tab=readme-ov-file#about-the-buckeye-data-splits](https://github.com/kamperh/vqwordseg?tab=readme-ov-file#about-the-buckeye-data-splits). Alignments are found here [https://github.com/kamperh/vqwordseg/releases/tag/v1.0](https://github.com/kamperh/vqwordseg/releases/tag/v1.0). Note that the JSON files found here [https://github.com/kamperh/zerospeech2021_baseline/tree/2f2c47766ffc02574dcc71fea7fe5247ca4f323c/datasets/buckeye](https://github.com/kamperh/zerospeech2021_baseline/tree/2f2c47766ffc02574dcc71fea7fe5247ca4f323c/datasets/buckeye) must be contained directory called beckeye_segments which is a sibling to the root BuckEye data directory.

**Example Usage**

    python3 preprocess_buckeye.py path/to/buckeye/data

### Preprocess ZRC2017

Python script name: `preprocess_zrc2017.py`

This script extracts individual wav files for the ZeroSpeech [https://download.zerospeech.com/](https://download.zerospeech.com/) 2017 corpus' train split based on the VAD files found in the test split's directory (where the alignment files can also be found) on the previously mentioned ZeroSpeech website. The language argument specifies the ZeroSpeech language to process, options are: english, french, mandarin, german, and wolof.

**ZeroSpeech Repository**

Clone the ZeroSpeech repository at [https://github.com/zerospeech/benchmarks](https://github.com/zerospeech/benchmarks) to use the ZeroSpeech toolkit used for benchmark resources and evaluation scripts.

**Example Usage**

    python3 preprocess_zrc2017.py path/to/zrc2017/data path/to/zrc2017/data/vad.vad.csv path/to/zrc2017/data/alignments.wrd --language=english

### Feature Extraction

To extract features, use a similar method as in [this](https://github.com/bshall/hubert/tree/main) repo:
1. Pad the waveform (frame center padding):
    `wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))`
2. Extract features, use:
  [this](https://github.com/bshall/hubert/blob/main/hubert/model.py#L39) `x, _ = model.encode(wav, layer=layer)`
  or, [this](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/hubert/hubert.py#L533)/[this](https://github.com/bshall/knn-vc/blob/848302a262f7299c738af49d74209790ed442a9f/wavlm/WavLM.py#L323) `x, _ = model.extract_features(wav, output_layer=layer)`

<!-- ### Extract Feature Encodings

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

The model_name can be one of: w2v2_fs, w2v2_hf, hubert_fs, hubert_hf, hubert_shall, melspec, mfcc. The optional extension argument is the extension of the audio files to be processed. -->

<!-- ### Sample and Transform Data

Python script name: audio_process.py

This script contains utility functions to sample audio (and its features), to normalize sampled features, to find corresponding alignment files, and to load the alignment file attributes. -->