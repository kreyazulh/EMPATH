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

def generate_new_list(sorted_numbers):
    new_list = []

    # Iterate through the sorted list
    i = 0
    while i < len(sorted_numbers):
        current_number = sorted_numbers[i]

        # Find the range for consecutive numbers
        start = current_number
        while i + 1 < len(sorted_numbers) and sorted_numbers[i + 1] == current_number + 1:
            current_number = sorted_numbers[i + 1]
            i += 1

        end = current_number

        # Add the range to the new list
        new_list.append([start - 1, end + 1])

        # Move to the next number in the sorted list
        i += 1

    return new_list

def find_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = list(set1.intersection(set2))
    return common_elements

def translate(point_cloud, translate_list):
    point_cloud_homogeneous = []
    for point in point_cloud:
        point_homogeneous = point.copy()
        point_homogeneous.append(1)
        point_cloud_homogeneous.append(point_homogeneous)

    # Define the translation
    tx = translate_list[0]
    ty = translate_list[1]
    tz = translate_list[2]

    # Construct the translation matrix
    translation_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [tx, ty, tz, 1],
    ]

    # Apply the transformation to our point cloud
    translated_points = np.matmul(
        point_cloud_homogeneous,
        translation_matrix)

    # Convert to cartesian coordinates
    translated_points_xyz = []
    for point in translated_points:
        point = np.array(point[:-1])
        translated_points_xyz.append(point)

    # Map original to translated point coordinates
    # (x0, y0, z0) → (x1, y1, z1)
    for i in range(len(point_cloud)):
        point = point_cloud[i]
        translated_point = translated_points_xyz[i]
        #print(f'{point} → {list(translated_point)}')
        return list(translated_point)


def get_values_optimized(file_name, xyz):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    cap = cv2.VideoCapture(file_name)

    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    res_mult = 1
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_right = 0
    max_left = 0
    count_right = 0
    count_left = 0
    right = 0
    left = 0
    right_hand_miss = []
    left_hand_miss = []
    i = 0
    j = 0
    video = []
    all_image = []
    results = None
    batch = 5
    fix = False
    next_fix = False
    frame = 0
    all_landmarks = []
    first_detect = False
    frame_count = 0

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        frame_count += 1
        # Skip the first 7 frames
        if frame_count <= 7:
            continue

        image = cv2.resize(image, (frame_width * res_mult, frame_height * res_mult))

        # results.face_landmarks diye coordinates access kora jabe (face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_landmarks)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        all_image.append([i, image, results])
        i = i + 1
        j = j + 1

        if (first_detect == False and results.left_hand_landmarks != None):
            first_detect = True

        if (i > 0 and i % batch == 0 and results.left_hand_landmarks != None and fix == False):
            fix = True
        elif (i > 0 and i % batch == 0 and results.left_hand_landmarks == None and fix == False):
            next_fix = True
        elif (next_fix == True) and results.left_hand_landmarks != None:
            fix = True
            next_fix = False
        elif (j == length):
            fix = True
            next_fix = False

        if (results.left_hand_landmarks == None):
            count_left = count_left + 1
            if (count_left > max_left):
                max_left = count_left
            if (first_detect == True):
                left_hand_miss.append(i)
                continue

        else:
            count_left = 0
            pose_x = results.pose_landmarks.landmark[15].x
            pose_y = results.pose_landmarks.landmark[15].y
            pose_z = results.pose_landmarks.landmark[15].z
            pose_list = [pose_x, pose_y, pose_z]

            left_hand_x = results.left_hand_landmarks.landmark[0].x
            left_hand_y = results.left_hand_landmarks.landmark[0].y
            left_hand_z = results.left_hand_landmarks.landmark[0].z
            left_hand_list = [left_hand_x, left_hand_y, left_hand_z]

            translate_x = left_hand_x - pose_x
            translate_y = left_hand_y - pose_y
            translate_z = left_hand_z - pose_z
            translate_list = [translate_x, translate_y, translate_z]
            new_pose_list = translate([pose_list], translate_list)

            setattr(results.pose_landmarks.landmark[15], 'x', new_pose_list[0])
            setattr(results.pose_landmarks.landmark[15], 'y', new_pose_list[1])
            setattr(results.pose_landmarks.landmark[15], 'z', new_pose_list[2])

            setattr(results.pose_landmarks.landmark[17], 'x', results.left_hand_landmarks.landmark[17].x)
            setattr(results.pose_landmarks.landmark[17], 'y', results.left_hand_landmarks.landmark[17].y)
            setattr(results.pose_landmarks.landmark[17], 'z', results.left_hand_landmarks.landmark[17].z)

            setattr(results.pose_landmarks.landmark[19], 'x', results.left_hand_landmarks.landmark[5].x)
            setattr(results.pose_landmarks.landmark[19], 'y', results.left_hand_landmarks.landmark[5].y)
            setattr(results.pose_landmarks.landmark[19], 'z', results.left_hand_landmarks.landmark[5].z)

            setattr(results.pose_landmarks.landmark[21], 'x', results.left_hand_landmarks.landmark[9].x)
            setattr(results.pose_landmarks.landmark[21], 'y', results.left_hand_landmarks.landmark[9].y)
            setattr(results.pose_landmarks.landmark[21], 'z', results.left_hand_landmarks.landmark[9].z)

        # for face (FACEMESH_TESSELATION, )FACEMESH_CONTOURS)
        # mp_holistic.FACE_CONNECTIONS (which joints connect which)
        # we will only focus on left hand for this case, as these signs are mostly single-handed

        if (fix == True):
            fix_left_hand_miss = generate_new_list(left_hand_miss)

            for j in range(len(fix_left_hand_miss)):
                start = fix_left_hand_miss[j][0]
                end = fix_left_hand_miss[j][1]
                multiplier = 1;
                for k in range(start + 1, end):
                    all_image[k - 1][2].left_hand_landmarks = copy.deepcopy(results.left_hand_landmarks)
                    for landmarks in range(len(results.left_hand_landmarks.landmark)):
                        # print(landmarks)
                        if (all_image[start - 1][2].left_hand_landmarks == None):
                            l = 2
                        else:
                            l = 1
                        # print("start")
                        # print(all_image[start-l][2].left_hand_landmarks.landmark[landmarks].x)
                        # print("end")
                        # print(all_image[end-1][2].left_hand_landmarks.landmark[landmarks].x)
                        # print("average")
                        # print((all_image[start-l][2].left_hand_landmarks.landmark[landmarks].x+(all_image[end-1][2].left_hand_landmarks.landmark[landmarks].x-all_image[start-l][2].left_hand_landmarks.landmark[landmarks].x)*(multiplier/(end-start))))

                        setattr(all_image[k - 1][2].left_hand_landmarks.landmark[landmarks], 'x', (
                                    all_image[start][2].left_hand_landmarks.landmark[landmarks].x + (
                                        all_image[end - l][2].left_hand_landmarks.landmark[landmarks].x -
                                        all_image[start][2].left_hand_landmarks.landmark[landmarks].x) * (
                                                multiplier / (end - start))))
                        setattr(all_image[k - 1][2].left_hand_landmarks.landmark[landmarks], 'y', (
                                    all_image[start][2].left_hand_landmarks.landmark[landmarks].y + (
                                        all_image[end - l][2].left_hand_landmarks.landmark[landmarks].y -
                                        all_image[start][2].left_hand_landmarks.landmark[landmarks].y) * (
                                                multiplier / (end - start))))
                        setattr(all_image[k - 1][2].left_hand_landmarks.landmark[landmarks], 'z', (
                                    all_image[start][2].left_hand_landmarks.landmark[landmarks].z + (
                                        all_image[end - l][2].left_hand_landmarks.landmark[landmarks].z -
                                        all_image[start][2].left_hand_landmarks.landmark[landmarks].z) * (
                                                multiplier / (end - start))))
                    multiplier = multiplier + 1

                    left_hand_miss = []
                    if all_image[k - 1][2].pose_landmarks != None:
                        pose_x = all_image[k - 1][2].pose_landmarks.landmark[15].x
                        pose_y = all_image[k - 1][2].pose_landmarks.landmark[15].y
                        pose_z = all_image[k - 1][2].pose_landmarks.landmark[15].z
                        pose_list = [pose_x, pose_y, pose_z]

                        left_hand_x = all_image[k - 1][2].left_hand_landmarks.landmark[0].x
                        left_hand_y = all_image[k - 1][2].left_hand_landmarks.landmark[0].y
                        left_hand_z = all_image[k - 1][2].left_hand_landmarks.landmark[0].z
                        left_hand_list = [left_hand_x, left_hand_y, left_hand_z]

                        translate_x = left_hand_x - pose_x
                        translate_y = left_hand_y - pose_y
                        translate_z = left_hand_z - pose_z
                        translate_list = [translate_x, translate_y, translate_z]
                        new_pose_list = translate([pose_list], translate_list)

                        setattr(all_image[k - 1][2].pose_landmarks.landmark[15], 'x', new_pose_list[0])
                        setattr(all_image[k - 1][2].pose_landmarks.landmark[15], 'y', new_pose_list[1])
                        setattr(all_image[k - 1][2].pose_landmarks.landmark[15], 'z', new_pose_list[2])

                        setattr(all_image[k - 1][2].pose_landmarks.landmark[17], 'x',
                                all_image[k - 1][2].left_hand_landmarks.landmark[17].x)
                        setattr(all_image[k - 1][2].pose_landmarks.landmark[17], 'y',
                                all_image[k - 1][2].left_hand_landmarks.landmark[17].y)
                        setattr(all_image[k - 1][2].pose_landmarks.landmark[17], 'z',
                                all_image[k - 1][2].left_hand_landmarks.landmark[17].z)

                        setattr(all_image[k - 1][2].pose_landmarks.landmark[19], 'x',
                                all_image[k - 1][2].left_hand_landmarks.landmark[5].x)
                        setattr(all_image[k - 1][2].pose_landmarks.landmark[19], 'y',
                                all_image[k - 1][2].left_hand_landmarks.landmark[5].y)
                        setattr(all_image[k - 1][2].pose_landmarks.landmark[19], 'z',
                                all_image[k - 1][2].left_hand_landmarks.landmark[5].z)

                        setattr(all_image[k - 1][2].pose_landmarks.landmark[21], 'x',
                                all_image[k - 1][2].left_hand_landmarks.landmark[9].x)
                        setattr(all_image[k - 1][2].pose_landmarks.landmark[21], 'y',
                                all_image[k - 1][2].left_hand_landmarks.landmark[9].y)
                        setattr(all_image[k - 1][2].pose_landmarks.landmark[21], 'z',
                                all_image[k - 1][2].left_hand_landmarks.landmark[9].z)

                    all_image[k - 1][1].flags.writeable = True
                    all_image[k - 1][1] = cv2.cvtColor(all_image[k - 1][1], cv2.COLOR_RGB2BGR)


            fix = False

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.append([i, image])

    for frame_data in sorted(all_image, key=lambda x: x[0]):
        landmarks = create_landmarks_df(frame_data[2], frame, xyz)
        all_landmarks.append(landmarks)

    holistic.close()
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

    return all_landmarks

train = pd.read_csv('train_gislr.csv')
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
    landmarks = get_values_optimized(video_file, xyz)

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



