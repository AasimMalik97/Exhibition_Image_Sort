from collections import defaultdict
import heapq
from multiprocessing import Pool, cpu_count
import itertools
import time

# Function to load data from a text file into a nested dictionary
def load_txt_to_dict(path):
    data = defaultdict(dict)  # Store rows grouped by index
    with open(path, 'r') as file:
        file.readline()  # Skip the first line (row count)
        for index, line in enumerate(file):
            row = line.strip().split()  # Split the line by whitespace
            type_key = row[0]  # The type of element ('P' or 'L')
            characteristics = set(row[2:])  # Store characteristics as a set
            data[index] = {"type": type_key, "characteristics": characteristics}
    return data

# Function to create frames optimized for large datasets
def create_frames(data_dict):
    frames = {}
    frame_index = 0
    used_keys = set()

    # Separate 'P' and 'L' types
    p_elements = {key: value for key, value in data_dict.items() if value['type'] == 'P'}
    l_elements = {key: value for key, value in data_dict.items() if value['type'] == 'L'}

    # Pair 'P' elements using a greedy approach with reduced comparisons
    while p_elements:
        key, row = p_elements.popitem()  # Take an arbitrary 'P' element
        if key in used_keys:
            continue

        # Select the best pair with minimal characteristics overlap
        best_pair = None
        min_common_chars = float('inf')

        for other_key in list(p_elements.keys()):  # Iterate through remaining 'P' elements
            if other_key in used_keys:
                continue
            common_chars = len(row['characteristics'] & p_elements[other_key]['characteristics'])
            if common_chars < min_common_chars:
                min_common_chars = common_chars
                best_pair = other_key

        if best_pair is not None:
            frames[frame_index] = [key, best_pair]
            used_keys.update([key, best_pair])
            p_elements.pop(best_pair)  # Remove the paired element
        else:
            frames[frame_index] = [key]  # Add as a single frame
            used_keys.add(key)

        frame_index += 1

    # Add 'L' type elements as individual frames
    for key in l_elements.keys():
        if key not in used_keys:
            frames[frame_index] = [key]
            frame_index += 1

    return frames

# Function to reorder frames optimized for large datasets
def reorder_frames(frames, data_dict):
    # Helper to calculate characteristics of a frame
    def get_frame_characteristics(frame):
        characteristics = set()
        for key in frame:
            characteristics.update(data_dict[key]["characteristics"])
        return characteristics

    # Precompute frame characteristics
    frame_characteristics = {
        frame_index: get_frame_characteristics(frame)
        for frame_index, frame in frames.items()
    }

    # Start with an arbitrary frame
    frame_indices = list(frames.keys())
    ordered_frames = [frame_indices.pop(0)]
    remaining_frames = set(frame_indices)

    # Reorder frames using a heuristic for maximum similarity
    while remaining_frames:
        current_frame = ordered_frames[-1]
        best_next_frame = None
        max_similarity = -1

        for candidate_frame in remaining_frames:
            similarity = len(
                frame_characteristics[current_frame] & frame_characteristics[candidate_frame]
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_next_frame = candidate_frame

        ordered_frames.append(best_next_frame)
        remaining_frames.remove(best_next_frame)

    # Rebuild the reordered frames dictionary
    return {index: frames[frame] for index, frame in enumerate(ordered_frames)}

# Function to write frames to an output file
def write_frames_to_file(frames, output_path):
    with open(output_path, 'w') as file:
        file.write(f"{len(frames)}\n")  # Write the number of frames
        for frame_index, frame in frames.items():
            frame_keys = " ".join(map(str, frame))
            file.write(frame_keys + "\n")

# Example usage
file_path1 = '../data/1_binary_landscapes.txt'
file_path2 = '../data/10_computable_moments.txt'
file_path3 = '../data/11_randomizing_paintings.txt'
file_path4 = '../data/110_oily_portraits.txt'
file_path5 = '../data/11_30000.txt' #30000
file_path6 = '../data/1000.txt' #1000
file_path7 = '../data/2000.txt' #2000
file_path8 = '../data/4000.txt' #4000
file_path9 = '../data/15000.txt' #15000



output_path = '../data/output_frames.txt'  # Path to the output file


# Load data from the input file
data_dict = load_txt_to_dict(file_path2)

# Create frames from the loaded data
start= time.time()
frames = create_frames(data_dict)
end = time.time()
# Reorder the frames for maximum common characteristics
start2= time.time()
reordered_frames = reorder_frames(frames, data_dict)
end2 = time.time()

# Write the reordered frames to the output file
write_frames_to_file(reordered_frames, output_path)



print("Execution time of the create frames is- ", end-start)
print("Execution time of the reorder frames is- ", end2-start2)

print("Execution total time is- ", end2-start)
