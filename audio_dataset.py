import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
import torch
import time
from scipy.signal import convolve
overall_start = time.time()

class AudioDataset(Dataset):

    def __init__(self, flac_folder, labels_file, deg, imp, device, runType, transform=None):
        self.flac_folder = flac_folder
        self.runType = runType
        if runType != 'Evaluate':
            self.labels = self.load_labels(labels_file, runType)
        self.transform = transform
        self.device = device
        self.filenames = [f for f in os.listdir(flac_folder) if f.endswith(".flac")]
        self.deg = deg
        self.imp = imp

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        audioname = filename.split('.')[0]
        #print(audioname)
        file_path = os.path.join(self.flac_folder, filename)
        
        # Load the audio data
        audio_data, _ = librosa.load(file_path, sr=None)  # sr=None keeps original sampling rate
        # Extract features (chromatic derivatives)
        #print('Extracting Features')
        features = self.feature_extraction(audio_data)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        # Get the label
        if self.runType != 'Evaluate':
            label = self.labels.get(audioname, None)
            if label is None:
                raise ValueError(f"Label not found for {audioname}")
            label_tensor = torch.tensor(label, dtype=torch.float32)

            sample = {'features': features_tensor, 'label': label_tensor}
            #print('Got sample')
            if self.transform:
                sample = self.transform(sample)

        else:
            sample = {'features': features_tensor}
            if self.transform:
                sample = self.transform(sample)
        #print(type(sample)) 
        return sample

    def feature_extraction(self, audio_data):
        #print('Computing Feature Extraction')
        chroma_features = self.get_CDs(audio_data)
        return chroma_features

    def get_CDs(self, sig_sec):

        lh = len(self.imp)
        
        sig_in = np.concatenate([np.zeros(lh // 2 + 1), sig_sec, np.zeros(lh // 2 + 1)])
        
        CD = np.zeros((len(sig_in) + len(self.imp) - 1, self.deg + 1))
        for k in range(self.deg + 1):
            CD[:, k] = convolve(sig_in, self.imp[:, k])

        CD = CD[lh:-lh, :]
        CD = CD[::24]
            
        return CD

    # def get_CDs(self, sig_sec):
    #     #print('Getting CDs')
    #     lh = len(self.imp)
    #     sig_in = np.concatenate([np.zeros(lh // 2 + 1), sig_sec, np.zeros(lh // 2 + 1)])
    #     CD = np.zeros((len(sig_in) + len(self.imp) - 1, self.deg + 1))

    #     def compute_convolution(k):
    #         return convolve(sig_in, self.imp[:, k])

    #     with ThreadPoolExecutor() as executor:
    #         results = list(executor.map(compute_convolution, range(self.deg + 1)))

    #     for k in range(self.deg + 1):
    #         CD[:, k] = results[k]

    #     CD = CD[lh:-lh, :]
    #     CD = CD[::24]
    #     return CD

    def load_labels(self, labels_file, runType):
        labels = {}
        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                filename = parts[0].split('.')[0]
                label = 1 if parts[1] == 'spoof' else 0  # 1 for spoof, 0 for bonafide
                labels[filename] = label
        return labels
