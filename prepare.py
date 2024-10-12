import numpy as np
import pickle
import os
from scipy.signal import resample
from sklearn.model_selection import train_test_split

# Load the dataset for each subject with latin1 encoding
def load_subject_data(subject_path):
    with open(subject_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

# Downsample the PPG signal from 64Hz to 40Hz
def downsample_signal(ppg_signal, original_fs=64, target_fs=40):
    num_samples = int(len(ppg_signal) * target_fs / original_fs)
    return resample(ppg_signal, num_samples)

# Function to segment the signal with an 8s window and 2s overlap
def segment_signal(signal, window_size, overlap_size):
    num_segments = (len(signal) - window_size) // overlap_size + 1
    segments = []
    for i in range(num_segments):
        start = i * overlap_size
        end = start + window_size
        segments.append(signal[start:end])
    return np.array(segments)

# Prepare X (PPG signal) and y (HR values) for training
def prepare_data(data, window_duration=8, overlap_duration=2, original_fs=64, target_fs=40):
    # Extract PPG signal and HR ground truth from the dataset
    ppg_signal = data['signal']['wrist']['BVP']
    hr_labels = data['label']

    # Downsample the PPG signal from 64Hz to 40Hz
    downsampled_ppg_signal = downsample_signal(ppg_signal, original_fs, target_fs)

    # Calculate the new window and overlap sizes in terms of samples
    window_size = window_duration * target_fs
    overlap_size = overlap_duration * target_fs

    # Segment the downsampled PPG signal
    X = segment_signal(downsampled_ppg_signal, window_size, overlap_size)

    # Align HR labels with the segments
    y = hr_labels[:len(X)]  # Make sure the length of y matches X

    return X, y

# Process all subjects in the directory
def process_all_subjects(subjects_dir):
    all_X = []
    all_y = []

    # Loop through each subject folder (S1, S2, ..., S15)
    for subject_folder in os.listdir(subjects_dir):
        subject_path = os.path.join(subjects_dir, subject_folder, f'{subject_folder}.pkl')

        if os.path.isfile(subject_path):
            print(f'Processing: {subject_path}')

            # Load the data for the subject
            subject_data = load_subject_data(subject_path)

            # Prepare the data
            X, y = prepare_data(subject_data)

            all_X.append(X)
            all_y.append(y)

    # Combine all subjects' data
    all_X = np.vstack(all_X)
    all_y = np.hstack(all_y)

    return all_X, all_y

## Replicate the 8-second segments to form 30-second segments
#def replicate_to_30_seconds(X, original_duration=8, target_duration=30, target_fs=40):
#    target_samples = target_duration * target_fs
#    replicated_X = []
#
#    # Replicate each 8-second segment to fill 30 seconds
#    for segment in X:
#        replicated_segment = np.tile(segment, (target_samples // len(segment) + 1,))[:target_samples]
#        replicated_X.append(replicated_segment)
#
#    return np.array(replicated_X)

# Directory containing all subject folders (S1, S2, ..., S15)
subjects_dir = './'  # Replace with the actual path

# Process the data for all subjects
X, y = process_all_subjects(subjects_dir)

# Replicate X (from 8s segments to 30s segments)
#X_replicated = replicate_to_30_seconds(X)

X_replicated = X

# Randomly divide the data into an 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X_replicated, y, test_size=0.2, random_state=42)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')

np.save('X_DALIA_train.npy', X_train)
np.save('X_DALIA_test.npy', X_test)
np.save('y_DALIA_train.npy', y_train)
np.save('y_DALIA_test.npy', y_test)

'''
Processing: ./S1/S1.pkl
Processing: ./S6/S6.pkl
Processing: ./S10/S10.pkl
Processing: ./S14/S14.pkl
Processing: ./S2/S2.pkl
Processing: ./S8/S8.pkl
Processing: ./S5/S5.pkl
Processing: ./S13/S13.pkl
Processing: ./S11/S11.pkl
Processing: ./S7/S7.pkl
Processing: ./S12/S12.pkl
Processing: ./S4/S4.pkl
Processing: ./S9/S9.pkl
Processing: ./S3/S3.pkl
Processing: ./S15/S15.pkl
Shape of X_train: (51757, 320)
Shape of X_test: (12940, 320)
Shape of y_train: (51757,)
Shape of y_test: (12940,)
'''
