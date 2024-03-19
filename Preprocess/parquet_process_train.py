import os
import pandas as pd

# Base directory for processed train files
base_dir = 'include_processed50/train'

# Initialize a list to hold data for the DataFrame
data = []

# Initialize participant ID
participant_id = 1
# Counter for assigning participant ID to every 7 videos
video_counter = 0

# Traverse the directory structure
for sign in os.listdir(base_dir):
    sign_dir = os.path.join(base_dir, sign)
    if os.path.isdir(sign_dir):
        for file in os.listdir(sign_dir):
            if file.endswith('.parquet'):
                # Construct the full path to the file
                file_path = os.path.join(sign_dir, file)
                # Append the data
                data.append({'path': file_path, 'participant_id': participant_id, 'sign': sign})
                video_counter += 1
                # Update participant ID every 11 videos
                if video_counter % 11 == 0:
                    participant_id += 1

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True)

# Write the DataFrame to a CSV file
train_csv_path = 'train.csv'
df.to_csv(train_csv_path, index=False)

print(f"CSV file '{train_csv_path}' has been created with shuffled rows.")
