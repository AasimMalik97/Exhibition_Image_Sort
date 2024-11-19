from collections import defaultdict

# Function to load data from a text file into a nested dictionary
def load_txt_to_dict(path):
    data = defaultdict(list)  # defaultdict to handle duplicate keys
    with open(path, 'r') as file:
        num_rows = int(file.readline().strip())  # Read and skip the first line
        for index, line in enumerate(file):
            row = line.strip().split()  # Split by whitespace
            key = index  # Use the index as the key
            type_key = row[0]  # First column as the type key
            num_characteristics = row[1]
            characteristics = row[2:]

            # Append each row's data under the same key
            data[key].append({
                "type": type_key,
                "num_characteristics": num_characteristics,
                "characteristics": characteristics
            })

    return data

# Function to create frames from the data
def create_frames(data_dict):
    frames = {}
    frame_index = 0
    used_keys = set()


    for key, value in data_dict.items():
        if key in used_keys:
            continue
        row = value[0]
        if row['type'] == 'P':
            # Try to find another 'P' row to pair with that has the minimum common characteristics
            min_common_chars = float('inf')
            best_pair = None
            for other_key, other_value in data_dict.items():
                if other_key != key and other_key not in used_keys and other_value[0]['type'] == 'P':
                    common_chars = len(set(row['characteristics']) & set(other_value[0]['characteristics']))
                    if common_chars < min_common_chars:
                        min_common_chars = common_chars
                        best_pair = other_key
            if best_pair is not None:
                frames[frame_index] = [key, best_pair]
                used_keys.update([key, best_pair])
                frame_index += 1
            else:
                # If no pair found, add the single 'P' row as a frame
                frames[frame_index] = [key]
                used_keys.add(key)
                frame_index += 1
        else:
            # Add 'L' row as a frame
            frames[frame_index] = [key]
            used_keys.add(key)
            frame_index += 1

    return frames

# Function to write frames to an output file
def write_frames_to_file(frames, output_path):
    with open(output_path, 'w') as file:
        file.write(f"{len(frames)}\n")  # Write the number of frames
        for frame_index, frame in frames.items():
            frame_keys = " ".join(map(str, frame))
            file.write(f"Frame {frame_index}: {frame_keys}\n")

# Example usage
file_path = '../data/11_randomizing_paintings.txt'
file_path2 = '../data/10_computable_moments.txt'
file_path3 = '../data/110_oily_portraits.txt'
output_path = '../data/output_frames.txt'
data_dict = load_txt_to_dict(file_path3)
frames = create_frames(data_dict)
write_frames_to_file(frames, output_path)