# Data Process

This repository is responsible for the preprocessing of speech data. This enables lightweight audio encodings involved in downstream tasks.

## Scripts

Note: all alignment and VAD files are saved in seconds and all file names containing time stamps are denoted in milliseconds.

### Feature Extraction

Python script name: `extract_features.py`

This script extracts and saves self-supervised speech features for an input dataset. Frame centering padding is always applied to the input waveforms: `wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))`.

**Example Usage**

    python3 extract_features.py model_name model_layer path/to/input/data path/to/output/directory --extension=.flac --layer_norm=False

The `model_name` argument specifies the SSL model used (as shown below).
The `model_layer` argument specifies the encoder layer to extract from: `1-12/24` or `0` for CNN output, `None` for normalized final layer output, or `-1` for all layer outputs (`0-12/24`).
The path to the input data and the output directory is provided.
The optional `extension` argument specifies the waveform format.
The `layer_norm` argument performs waveform normalization.

The models currently included are: [wav2vec2](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) (base: `w2v2` and large: `w2v2_large`), [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) (base: `hubert` and large: `hubert_large`), [HuBERT-Soft](https://github.com/bshall/hubert) (encoder: `hubert_soft_enc` and post-projection features: `hubert_soft`), and [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) (base: `wavlm` and large `wavlm_large`).

### Feature Slicing

Python script name: `feature_slicing.py`

This script slices speech features and alignment files (mainly used for the LibriSpeech corpus [https://www.openslr.org/12](https://www.openslr.org/12)) based on the silences found in the alignments ([https://zenodo.org/records/2619474](https://zenodo.org/records/2619474)). Syllable alignments are created and added to the TextGrid files (requires the repo at [https://github.com/kylebgorman/syllabify](https://github.com/kylebgorman/syllabify)).

**Example Usage**

    python3 feature_slicing.py path/to/alignment/data --features_dir=path/to/speech/features --extract_grid
  
This script assumes a similar path structure for features `/.../.../librispeech/dev_clean/model/layer`, and for alignments `/.../.../librispeech_alignments/dev_clean`. `extract_grid` (default: `True`) determines if alignments are split or not, the presence of `features_dir` determines if features are split or not.

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