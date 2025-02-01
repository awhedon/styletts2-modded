#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        # Set root_path first before filtering
        self.root_path = root_path
        self.sr = sr

        _data_list = [l.strip().split('|') for l in data_list]
        _processed_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        # Filter valid files during initialization
        self.data_list = self._filter_valid_files(_processed_list)
        if len(self.data_list) == 0:
            raise RuntimeError("No valid files found after filtering!")
        self.text_cleaner = TextCleaner()

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
    def _filter_valid_files(self, data_list):
        """Filter out invalid or too short audio files."""
        valid_files = []
        print(f"Filtering {len(data_list)} files...")
        
        for data in data_list:
            path = data[0]
            full_path = osp.join(self.root_path, path)
            
            try:
                # Check if file exists
                if not osp.exists(full_path):
                    alt_path1 = osp.join(self.root_path, "audio_data", path)
                    alt_path2 = osp.join(self.root_path, path.replace("audio_data/", ""))
                    if osp.exists(alt_path1):
                        full_path = alt_path1
                    elif osp.exists(alt_path2):
                        full_path = alt_path2
                    else:
                        print(f"Skipping {path}: File not found")
                        continue
                
                # Try to load the audio
                audio, sr = librosa.load(full_path, sr=None)
                duration = len(audio) / sr
                
                # Check duration
                if duration < 1.0:  # Minimum 1 second duration
                    print(f"Skipping {path}: Too short ({duration:.2f}s)")
                    continue
                
                valid_files.append(data)
                
            except Exception as e:
                print(f"Skipping {path}: {str(e)}")
                continue
        
        print(f"Found {len(valid_files)} valid files out of {len(data_list)}")
        return valid_files

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 10  # Maximum number of attempts to get valid data
        attempts = 0
        
        while attempts < max_attempts:
            try:
                data = self.data_list[idx]
                path = data[0]
                
                # Load tensor with error handling
                try:
                    wave, text_tensor, speaker_id = self._load_tensor(data)
                    if wave is None:  # If _load_tensor failed
                        idx = (idx + 1) % len(self.data_list)
                        attempts += 1
                        continue
                except Exception as e:
                    print(f"Error loading tensor for {path}: {str(e)}")
                    idx = (idx + 1) % len(self.data_list)
                    attempts += 1
                    continue
                
                try:
                    mel_tensor = preprocess(wave).squeeze()
                except Exception as e:
                    print(f"Error in preprocess for {path}: {str(e)}")
                    idx = (idx + 1) % len(self.data_list)
                    attempts += 1
                    continue
                
                acoustic_feature = mel_tensor.squeeze()
                length_feature = acoustic_feature.size(1)
                acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
                
                # Get reference sample with error handling
                try:
                    ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
                    ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
                    if ref_mel_tensor is None:  # Check if reference loading failed
                        idx = (idx + 1) % len(self.data_list)
                        attempts += 1
                        continue
                except Exception as e:
                    print(f"Error loading reference data for {path}: {str(e)}")
                    idx = (idx + 1) % len(self.data_list)
                    attempts += 1
                    continue
                
                # Get OOD text with error handling
                try:
                    ps = ""
                    text_attempts = 5  # Prevent infinite loop
                    text_try = 0
                    
                    while len(ps) < self.min_length and text_try < text_attempts:
                        rand_idx = np.random.randint(0, len(self.ptexts) - 1)
                        ps = self.ptexts[rand_idx]
                        text_try += 1
                    
                    if text_try >= text_attempts:
                        print(f"Warning: Could not find text of sufficient length for {path}")
                        ps = self.ptexts[0]  # Use first text as fallback
                        
                    text = self.text_cleaner(ps)
                    text.insert(0, 0)
                    text.append(0)
                    ref_text = torch.LongTensor(text)
                    
                except Exception as e:
                    print(f"Error processing OOD text for {path}: {str(e)}")
                    idx = (idx + 1) % len(self.data_list)
                    attempts += 1
                    continue
                
                return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave
                
            except Exception as e:
                print(f"Error in __getitem__ for index {idx}: {str(e)}")
                idx = (idx + 1) % len(self.data_list)
                attempts += 1
                continue
        
        raise RuntimeError(f"Failed to load valid data after {max_attempts} attempts starting from index {idx}")

    def _load_tensor(self, data):
        try:
            wave_path, text, speaker_id = data
            speaker_id = int(speaker_id)
            full_path = osp.join(self.root_path, wave_path)
            
            # Debug prints
            print(f"Root path: {self.root_path}")
            print(f"Wave path: {wave_path}")
            print(f"Full path: {full_path}")
            print(f"Current working directory: {os.getcwd()}")
            
            if not osp.exists(full_path):
                # Try alternative paths
                alt_path1 = osp.join(self.root_path, "audio_data", wave_path)
                alt_path2 = osp.join(self.root_path, wave_path.replace("audio_data/", ""))
                alt_path3 = wave_path
                
                print(f"Trying alternative paths:")
                print(f"Alt1: {alt_path1} exists: {osp.exists(alt_path1)}")
                print(f"Alt2: {alt_path2} exists: {osp.exists(alt_path2)}")
                print(f"Alt3: {alt_path3} exists: {osp.exists(alt_path3)}")
                
                if osp.exists(alt_path1):
                    full_path = alt_path1
                elif osp.exists(alt_path2):
                    full_path = alt_path2
                elif osp.exists(alt_path3):
                    full_path = alt_path3
                else:
                    raise FileNotFoundError(f"Could not find audio file in any expected location")
            
            try:
                wave, sr = librosa.load(full_path, sr=None)
            except Exception as e:
                print(f"Error loading audio with librosa: {str(e)}")
                # Try alternative loading method or skip file
                return None

            # Check duration
            duration = len(wave) / sr
            min_duration = 1.0  # minimum 1 second
            if duration < min_duration:
                print(f"Audio too short ({duration:.2f}s): {wave_path}")
                return None
                
            if len(wave.shape) > 1:  # If stereo
                wave = wave[:, 0].squeeze()
                
            if sr != 24000:
                try:
                    wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
                    print(f"Resampled {wave_path} from {sr} to 24000")
                except Exception as e:
                    print(f"Error resampling: {str(e)}")
                    return None
                    
            wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
            
            text = self.text_cleaner(text)
            text.insert(0, 0)
            text.append(0)
            text = torch.LongTensor(text)
    
            return wave, text, speaker_id
            
        except Exception as e:
            print(f"Error in _load_tensor for {wave_path}: {str(e)}")
            print(f"Data: {data}")
            return None

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels



def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

