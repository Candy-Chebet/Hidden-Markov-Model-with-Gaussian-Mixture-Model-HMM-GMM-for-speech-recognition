from audio_generator import AudioGenerator

def main():
    # Initialize the AudioGenerator and load data
    audio_gen = AudioGenerator(spectrogram=True)
    audio_gen.load_train_data()

    # Train the HMM-GMM model
    num_states = 10  # You can adjust the number of states and mixtures based on your data
    num_mixtures = 5
    audio_gen.train_hmm_gmm(num_states=num_states, num_mixtures=num_mixtures, max_iterations=100)

    # Recognize speech using the HMM-GMM model
    test_audio_path = 'path/to/test_audio.wav'
    recognized_text = audio_gen.recognize_hmm_gmm(test_audio_path)
    print('HMM-GMM Recognized Text:', recognized_text)

    


if __name__ == "__main__":
    main()
