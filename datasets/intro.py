import librosa
from functools import lru_cache
import torch
import warnings
import pytorch_lightning as L
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from numpy.lib.stride_tricks import as_strided

def sliding_window_view_with_hop(array: np.ndarray, window_size: int, hop: int) -> np.ndarray:
    assert array.ndim == 1
    size = array.shape[0]
    stride = array.strides[0]
    return as_strided(array,
                      shape=((int((size - window_size) / hop) + 1), window_size),
                      strides=(stride * hop, stride))

def get_batch_count(duration, window_width, hop):
    return max(0, np.ceil((duration - window_width) / hop)) + 1
    

SAMPLE_RATE = 16000

class IntroDataset(Dataset):
    # TODO: Change this to a IterableDataset
    # Since we create multiple data points from a single file, a implementation using iterable datasets can be more efficient
    # This will require implementing dataset splits into the dataset class itself, since random_split doesn't work with iterable datasets
    # Implementing splitting should also make it simpler to implement multithreaded dataloading
    def __init__(self, file_list_path, audio_dir_path, input_length, stft_hop_length, input_hop_length):
        
        self.audio_dir_path = Path(audio_dir_path)
        self.file_list = pd.read_csv(file_list_path)
        self.file_list['chunk_count'] = \
            self.file_list.intro_end.apply(lambda x: get_batch_count(x*2, input_length, input_hop_length))
        self.length = int(self.file_list.chunk_count.sum().item())

        self.input_length = input_length
        self.input_hop_length = input_hop_length

        self.fps = SAMPLE_RATE / (4 * stft_hop_length)
        self.frame_count = int(self.input_length * self.fps)

    def __len__(self):
        return self.length

    @lru_cache
    def load_batches_from_file(self, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav, _ = librosa.load(str(self.audio_dir_path / filename),
                                sr = SAMPLE_RATE,
                                mono = True)
        batches = sliding_window_view_with_hop(wav, int(self.input_length * SAMPLE_RATE), int(self.input_hop_length * SAMPLE_RATE))
        return torch.from_numpy(batches)

    def __getitem__(self, index):
        file_index = np.where(self.file_list.chunk_count.cumsum() > index)[0][0].item()
        file_start_chunk = int(self.file_list.chunk_count.cumsum()[file_index - 1].item()) if file_index > 0 else 0
        chunk_index = index - file_start_chunk
        file = self.file_list.iloc[file_index]

        batches = self.load_batches_from_file(file.filename)

        chunk_start_time = self.input_hop_length * chunk_index
        y = (torch.arange(self.frame_count) < (file.intro_end - chunk_start_time) * self.fps).type(torch.long)
        
        return batches[chunk_index], y

class IntroDataModule(L.LightningDataModule):
    def __init__(self, file_list_path, audio_dir_path, input_length, stft_hop_length, input_hop_length, batch_size, pin_memory):
        super().__init__()
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.file_list_path = file_list_path
        self.audio_dir_path = audio_dir_path
        self.input_length = input_length
        self.input_hop_length = input_hop_length
        self.stft_hop_length = stft_hop_length


    def setup(self, stage=None):
        self.dataset = IntroDataset(self.file_list_path, self.audio_dir_path, self.input_length, self.stft_hop_length, self.input_hop_length)
        # self.train_ds, self.val_ds, self.test_ds = random_split(self.dataset, [0.7, 0.15, 0.15])
        train_size = int(len(self.dataset) * 0.7)
        val_size = int(len(self.dataset) * 0.15)
        self.train_ds = Subset(self.dataset, range(0, train_size))
        self.val_ds = Subset(self.dataset, range(train_size, train_size+val_size))
        self.test_ds = Subset(self.dataset, range(train_size+val_size, len(self.dataset)))

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=1,
                          pin_memory=self.pin_memory)
                          
    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=1,
                          pin_memory=self.pin_memory)
