import json
import numpy as np
import random
from python_speech_features import mfcc
import librosa
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence
from utils import conv_output_length
from hmm_gmm import HMMGMM

RNG_SEED = 123

class AudioGenerator():
    def __init__(self, step=10, window=20, max_freq=8000, mfcc_dim=13,
                 minibatch_size=20, desc_file=None, spectrogram=True, max_duration=10.0, 
                 sort_by_duration=False):
        self.feat_dim = calc_feat_dim(window, max_freq)
        self.mfcc_dim = mfcc_dim
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.rng = random.Random(RNG_SEED)
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.cur_train_index = 0
        self.cur_valid_index = 0
        self.cur_test_index = 0
        self.max_duration = max_duration
        self.minibatch_size = minibatch_size
        self.spectrogram = spectrogram
        self.sort_by_duration = sort_by_duration

    def load_metadata_from_desc_file(self, desc_file, partition):
        """Load metadata (file paths and transcripts) from the description file for a given partition.
        Params:
            desc_file (str): Path to the JSON description file containing the metadata.
            partition (str): The partition for which to load the metadata (train, valid, or test).
        """
        with open(desc_file, 'r') as f:
            data = json.load(f)

        if partition not in data:
            raise ValueError(f"Partition '{partition}' not found in the description file.")

        metadata = data[partition]

        if not isinstance(metadata, list):
            raise ValueError("Metadata in the description file must be a list.")

        self.audio_paths = []
        self.transcripts = []

        for entry in metadata:
            if 'file_path' not in entry or 'transcript' not in entry:
                raise ValueError("Each entry in metadata must have 'file_path' and 'transcript' fields.")

            audio_path = entry['file_path']
            transcript = entry['transcript']

            if not isinstance(audio_path, str) or not isinstance(transcript, str):
                raise ValueError("Both 'file_path' and 'transcript' must be strings.")

            self.audio_paths.append(audio_path)
            self.transcripts.append(transcript)

        # If spectrogram is enabled, calculate the maximum duration for the given partition
        if self.spectrogram:
            max_duration_samples = int(self.max_duration * self.window * self.max_freq / 1000)
            self.audio_paths = [path for path in self.audio_paths if librosa.get_duration(filename=path) <= self.max_duration]
            self.transcripts, self.audio_paths = zip(*[(transcript, path) for transcript, path in zip(self.transcripts, self.audio_paths)
                                                       if len(mfcc_from_file(path, self.step, self.window, self.max_freq, numcep=self.mfcc_dim)) <= max_duration_samples])
        
        if self.sort_by_duration:
            self.sort_data_by_duration(partition)

    # Rest of the code (including get_batch, shuffle_data_by_partition, sort_data_by_duration, next_train, next_valid, next_test, load_train_data, load_validation_data, load_test_data, load_metadata_from_desc_file, fit_train, featurize, normalize)

    def train_hmm_gmm(self, num_states, num_mixtures, max_iterations=100):
        """ Train the HMM-GMM model on the training data
        Params:
            num_states (int): Number of states for the HMM
            num_mixtures (int): Number of Gaussian mixtures per state
            max_iterations (int): Maximum iterations for HMM-GMM training
        """
        # Get features and normalize them
        train_features = [self.normalize(self.featurize(a)) for a in self.audio_paths]
        train_features = np.vstack(train_features)

        # Initialize and train HMM-GMM
        hmm_gmm = HMMGMM(num_states, num_mixtures)
        hmm_gmm.train(train_features, max_iterations=max_iterations)

        # Save the HMM-GMM model
        self.hmm_gmm_model = hmm_gmm

    def recognize_hmm_gmm(self, audio_path):
        """ Recognize speech using the HMM-GMM model
        Params:
            audio_path (str): Path to the audio clip for recognition
        Returns:
            recognized_text (str): Recognized text from the audio
        """
        # Extract features and normalize
        features = self.normalize(self.featurize(audio_path))
        # Use the HMM-GMM model to recognize the speech
        recognized_text = self.hmm_gmm_model.recognize(features)
        return recognized_text
    
    def load_train_data(self, desc_file='train_corpus.json'):
        self.load_metadata_from_desc_file(desc_file, 'train')
        self.fit_train()
        if self.sort_by_duration:
            self.sort_data_by_duration('train')
