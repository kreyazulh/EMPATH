import matplotlib.pyplot as plt
SMALL_SIZE = 14
MEDIUM_SIZE = 15
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import pandas as pd
import os
import numpy as np
import copy
import mediapipe as mp
import sys
import cv2
import gc  # Import garbage collector

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def create_landmarks_df(results, frame, xyz):
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    face = pd.DataFrame()
    if results.face_landmarks is not None:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, 'x'] = point.x
            face.loc[i, 'y'] = point.y
            face.loc[i, 'z'] = point.z

    pose = pd.DataFrame()
    if results.pose_landmarks is not None:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, 'x'] = point.x
            pose.loc[i, 'y'] = point.y
            pose.loc[i, 'z'] = point.z

    left_hand = pd.DataFrame()
    if results.left_hand_landmarks is not None:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, 'x'] = point.x
            left_hand.loc[i, 'y'] = point.y
            left_hand.loc[i, 'z'] = point.z

    right_hand = pd.DataFrame()
    if results.right_hand_landmarks is not None:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, 'x'] = point.x
            right_hand.loc[i, 'y'] = point.y
            right_hand.loc[i, 'z'] = point.z

    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)

    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks


def get_values(file_name, xyz):
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Error opening video stream or file")
        raise IOError("Could not open video file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    res_mult = 1
    frame_count = 0  # Initialize frame count

    all_landmarks = []

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        frame_count += 1
        # Skip the first 7 frames
        if frame_count <= 7:
            continue

        image = cv2.resize(image, (frame_width * res_mult, frame_height * res_mult))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        landmarks = create_landmarks_df(results, frame_count, xyz)
        all_landmarks.append(landmarks)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    holistic.close()
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

    return all_landmarks

pq_file = 'demo.parquet'
xyz = pd.read_parquet(pq_file)

# Function to load checkpoints
def load_checkpoints(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            processed_videos = file.read().splitlines()
    else:
        processed_videos = []
    return set(processed_videos)


# Function to save checkpoints
def save_checkpoint(checkpoint_file, video_name):
    with open(checkpoint_file, 'a') as file:
        file.write(video_name + '\n')


# Assuming get_values is defined elsewhere and imported
# from your_module import get_values
# Assume xyz is defined or imported as well

# Path to the CSV file
csv_file = "file:///home/reyazul/Desktop/Thesis/New Research/video_details.csv"

# Base directory for processed files
base_processed_dir = "/home/reyazul/Desktop/Thesis/New Research/include50_processed"

# Path for the checkpoint file
checkpoint_file = os.path.join(base_processed_dir, 'checkpoints.txt')

# Ensure the base directories for train and test exist
train_dir = os.path.join(base_processed_dir, 'train')
test_dir = os.path.join(base_processed_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load processed videos from checkpoint
processed_videos = load_checkpoints(checkpoint_file)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Loop through the DataFrame
for index, row in df.iterrows():
    video_file = row['Video Path']  # Path to the video
    video_name = row['Video Name'].split('.')[0]  # Video name without extension

    # Skip this video if it's already processed
    if video_name in processed_videos:
        continue

    label = row['Label']  # Label of the video
    data_type = row['Type']  # Train or Test

    # Use your get_values function to process the video
    landmarks = get_values(video_file, xyz)

    # Assuming landmarks is a DataFrame or compatible format
    landmarks_df = pd.concat(landmarks).reset_index(drop=True)

    # Determine the correct directory based on type (train/test) and create label directory if it doesn't exist
    type_dir = os.path.join(base_processed_dir, data_type, label)
    os.makedirs(type_dir, exist_ok=True)

    # Save the landmarks DataFrame to a Parquet file in the correct directory
    parquet_path = os.path.join(type_dir, f"{video_name}.parquet")
    landmarks_df.to_parquet(parquet_path)

    # Save this video to the checkpoint
    save_checkpoint(checkpoint_file, video_name)

    # After saving the Parquet file, release memory
    del landmarks
    del landmarks_df
    gc.collect()  # Manually trigger garbage collection

print(
    "All videos have been processed and saved to Parquet files according to their labels and type (train/test). Checkpoints were used to track progress.")


