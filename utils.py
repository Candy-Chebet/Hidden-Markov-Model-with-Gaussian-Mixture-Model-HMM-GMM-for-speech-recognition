import json
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

def calc_feat_dim(window, max_freq):
    # Calculate feature dimension for given window size and max frequency
    return int(max_freq * window / 1000) + 1

def spectrogram_from_file(audio_clip, step, window, max_freq):
    # Implement the spectrogram extraction from the audio clip (similar to original code)
    pass

def text_to_int_sequence(text):
    # Implement a function to convert text to integer sequence (if needed for evaluation)
    pass

def conv_output_length(input_length, filter_size, stride, padding='valid'):
    """ Compute the output length after 1D convolutional layer
    Params:
        input_length (int): Length of the input sequence
        filter_size (int): Size of the filter/kernel
        stride (int): Stride length
        padding (str): Padding type ('valid' or 'same')
    Returns:
        int: Output length after convolution
    """
    if padding == 'valid':
        return np.ceil((input_length - filter_size + 1) / stride)
    elif padding == 'same':
        return np.ceil(input_length / stride)
    else:
        raise ValueError("Invalid padding type. Use 'valid' or 'same'.")
