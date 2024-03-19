import os
import pandas as pd

# Base directory for processed test files
#base_dir = 'processed_with_algo/test'
base_dir = 'include50_processed/test'

# Initialize a list to hold data for the DataFrame
data = []

# Traverse the directory structurea
for label in os.listdir(base_dir):
    label_dir = os.path.join(base_dir, label)
    if os.path.isdir(label_dir):
        for file in os.listdir(label_dir):
            if file.endswith('.parquet'):
                # Construct the full path to the file
                file_path = os.path.join(label_dir, file)
                # Append the data
                data.append({'path': file_path, 'sign': label})

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
test_csv_path = 'test.csv'
df.to_csv(test_csv_path, index=False)

print(f"CSV file '{test_csv_path}' has been created.")
