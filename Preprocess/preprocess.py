import csv
import re
import os
import glob

def find_video_file(folder_path, base_video_name):
    # Search for video files with any extension
    search_pattern = os.path.join(folder_path, base_video_name + '.*')
    for file in glob.glob(search_pattern):
        if file.endswith(('.mp4', '.MOV')):  # Considered video file extensions
            return os.path.basename(file)  # Return the file name with extension if found
    # If no file is found, assume it's an .mp4 file
    return base_video_name + '.MP4'  # Default to .mp4 extension

def process_paths(file_path, video_type):
    with open(file_path, 'r') as file:
        for line in file:
            original_path = line.strip()
            parts = original_path.split('/')
            folder_path = 'INCLUDE/' + '/'.join(parts[:-1])
            base_video_name, _ = os.path.splitext(parts[-1])  # Splits off the extension
            label = parts[-2]
            label_cleaned = re.sub(r'^\d+\.\s*', '', label)

            # Find the actual video file in the folder, defaulting to .mp4 if not found
            video_name_with_ext = find_video_file(folder_path, base_video_name)
            full_path = os.path.join(folder_path, video_name_with_ext)
            yield full_path, video_name_with_ext, label_cleaned, video_type

def write_csv():
    with open('video_details_full.csv', 'w', newline='') as csvfile:
        fieldnames = ['Video Path', 'Video Name', 'Label', 'Type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for path in ['INCLUDE/include_test.txt', 'INCLUDE/include_train.txt']:
            video_type = 'test' if 'test' in path else 'train'
            for video_path, video_name, label, video_type in process_paths(path, video_type):
                writer.writerow({'Video Path': video_path, 'Video Name': video_name, 'Label': label, 'Type': video_type})

write_csv()
