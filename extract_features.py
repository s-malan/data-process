"""
Encode audio data with different SSL models to be used by downstream tasks.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: November 2025
"""

import os
import argparse
from pathlib import Path
from typing import Union
from tqdm import tqdm

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F

import fairseq
from wavlm.WavLM import WavLM, WavLMConfig

class EncodeAudio:
    def __init__(
        self,
        model_name: str,
        layer: Union[str, None],
        data_dir: Path,
        save_dir: Path,
        extension: str,
        layer_norm: bool = False
    ):
        self.model_name = model_name
        self.layer = layer
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.extension = extension
        self.layer_norm = layer_norm
        ckpt_path = "checkpoints/"

        if self.model_name in ["w2v2", "w2v2_large"]:
            ckpt_path = ckpt_path + "wav2vec_small.pt" if self.model_name == "w2v2" else ckpt_path + "wav2vec_large.pt"
            models, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
            self.model = models[0].eval().cuda()
            self.cfg = cfg

        elif self.model_name in ["hubert", "hubert_large"]:
            ckpt_path = ckpt_path + "hubert_base_ls960.pt" if self.model_name == "hubert" else ckpt_path + "hubert_large_ll60k.pt"
            models, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
            self.model = models[0].eval().cuda()
            self.cfg = cfg

        elif self.model_name in ["hubert_soft_enc", "hubert_soft"]:
            self.model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda()

        elif self.model_name in ["wavlm", "wavlm_large"]:
            ckpt_path = ckpt_path + "WavLM-Base.pt" if self.model_name == "wavlm" else ckpt_path + "WavLM-Large.pt"
            ckpt = torch.load(ckpt_path)
            self.cfg = WavLMConfig(ckpt['cfg'])
            self.model = WavLM(self.cfg)
            self.model.load_state_dict(ckpt['model'])
            self.model.eval().cuda()

        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

    def save_encodings(self, wav: torch.Tensor, file_path: Path):
        """
        Call the appropriate encoding function based on the model type.

        Parameters
        ----------
        wav : torch.Tensor
            The audio waveform
        file_path : Path
            The path to the audio file
        """
        
        if self.model_name in ["w2v2", "w2v2_large"]:
            self.encode_w2v2(wav, file_path)
        elif self.model_name in ["hubert", "hubert_large"]:
            self.encode_hubert(wav, file_path)
        elif self.model_name in ["hubert_soft_enc", "hubert_soft"]:
            self.encode_hubert_shall(wav, file_path)
        elif self.model_name in ["wavlm", "wavlm_large"]:
            self.encode_wavlm(wav, file_path)
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")
    
    def save_encoding(self, x: torch.Tensor, file_path: Path, i: int):
        """
        Save the extracted features to disk.

        Parameters
        ----------
        x : torch.Tensor
            The extracted features
        file_path : Path
            The path to the audio file
        i : int
            The layer number
        """

        _, last_dir = os.path.split(self.save_dir)
        parts = str(file_path).split('/')
        copy_index = parts.index(last_dir)
        path_suffix = '/'.join(parts[copy_index + 1:])

        norm_str = "_layernorm" if self.layer_norm else ""
        out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i}{norm_str}')) / path_suffix
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())
    
    @torch.inference_mode()
    def encode_w2v2(self, wav: torch.Tensor, file_path: Path):
        """
        Extracts and saves wav2vec2 (Base or Large) features for a given audio file.

        Parameters
        ----------
        wav : torch.Tensor
            The audio waveform
        file_path : Path
            The path to the audio file
        """

        layer = np.arange(self.cfg.model.encoder_layers + 1) if self.layer == -1 else [self.layer]
        if layer[0] is None:
            print(f'Model {self.model_name} does not support layer norm. Setting to final layer instead.')
            layer = [self.cfg.model.encoder_layers]

        for i in layer:
            x = self.model.extract_features(wav, padding_mask=None, layer = i - 1)["x"]
            self.save_encoding(x, file_path, i)

    @torch.inference_mode()
    def encode_hubert(self, wav: torch.Tensor, file_path: Path):
        """
        Extracts and saves HuBERT (Base or Large) features for a given audio file.

        Parameters
        ----------
        wav : torch.Tensor
            The audio waveform
        file_path : Path
            The path to the audio file
        """

        layer = np.arange(self.cfg.encoder_layers + 1) if self.layer == -1 else [self.layer]
        if layer[0] is None:
            print(f'Model {self.model_name} does not support layer norm. Setting to final layer instead.')
            layer = [self.cfg.encoder_layers]

        for i in layer:
            x, _ = self.model.extract_features(wav, output_layer = i)
            self.save_encoding(x, file_path, i)
    
    @torch.inference_mode()
    def encode_hubert_shall(self, wav: torch.Tensor, file_path: Path):
        """
        Extracts and saves HuBERT-Soft (encoder or post-projection) features for a given audio file.

        Parameters
        ----------
        wav : torch.Tensor
            The audio waveform
        file_path : Path
            The path to the audio file
        """

        wav = wav.unsqueeze(0)
        layer = np.arange(self.model.encoder.layers.__len__() + 1) if self.layer == -1 else [self.layer]
        if layer[0] is None:
            print(f'Model {self.model_name} does not support layer norm. Setting to final layer instead.')
            layer = [self.model.encoder.layers.__len__()]

        for i in layer:
            x, _ = self.model.encode(wav, layer=i)
            if self.model_name == "hubert_soft":
                x = self.model.proj(x)
            self.save_encoding(x, file_path, i)

    @torch.inference_mode()
    def encode_wavlm(self, wav: torch.Tensor, file_path: Path):
        """
        Extracts and saves WavLM (Base or Large) features for a given audio file.

        Parameters
        ----------
        wav : torch.Tensor
            The audio waveform
        file_path : Path
            The path to the audio file
        """

        layer = np.arange(self.cfg.encoder_layers + 1) if self.layer == -1 else [self.layer]

        for i in layer:
            x, _ = self.model.extract_features(wav, output_layer=i)
            self.save_encoding(x, file_path, i)

    def extract_audio(self, file_path: str) -> torch.Tensor:
        """
        Load and pre-process audio file: apply layer normalization if specified, and pad for frame centering.

        Parameters
        ----------
        file_path : String
            The path to the audio file

        Returns
        -------
        wav : torch.Tensor
            The audio waveform
        """

        wav, sr = torchaudio.load(file_path, backend='soundfile')
        assert sr == 16000
        assert wav.ndim == 2
        assert wav.size(0) == 1

        if self.layer_norm:
            wav = torch.nn.functional.layer_norm(wav , wav.shape)

        if wav.shape[-1] < 400: # pad to be at least 400 (otherwise cannot encode)
            wav = F.pad(wav, (((400 - wav.shape[-1]) // 2), ((400 - wav.shape[-1]) // 2)))
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))

        return wav.cuda()
    
    def get_encodings(self):
        """
        Extract and save the encodings for the dataset.
        """

        for (dirpath, _, filenames) in tqdm(os.walk(self.data_dir)):
            if not filenames: # no files in directory
                continue
            
            # not in root of dataset path
            if dirpath is not self.data_dir:

                # walk through files in directory
                for file in tqdm(filenames):
                    if not file.endswith(self.extension): # ensure only audio files are processed
                        continue
                    
                    file_path = os.path.join(dirpath, file)
                    wav = self.extract_audio(file_path)
                    self.save_encodings(wav, Path(file_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "model",
        help="available models",
        type=str,
    )
    parser.add_argument(
        "layer",
        help="layer to extract from the model (-1 for all layers, None for layer_norm on final output).",
        type=str,
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    parser.add_argument(
        "--layer_norm",
        help="apply normalization to the waveform before encoding.",
        default=False,
        type=bool,
    )
    args = parser.parse_args()

    args.layer = int(args.layer) if args.layer != "None" else None
    if "large" in args.model and args.layer is not None:
        assert (
            args.layer <= 24 and args.layer >= -1
        ), "Layer must be between -1 and 24 (inclusive)."
    elif "large" not in args.model and args.layer is not None:
        assert (
            args.layer <= 12 and args.layer >= -1
        ), "Layer must be between -1 and 12 (inclusive)."

    encoder = EncodeAudio(args.model, args.layer, args.in_dir, args.out_dir, args.extension, args.layer_norm)
    encoder.get_encodings()