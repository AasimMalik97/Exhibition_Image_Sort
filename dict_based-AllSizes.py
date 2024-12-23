
from collections import defaultdict
import heapq
import numpy as np
import time

# Load data from file
def load_txt_to_dict(path):
    data = defaultdict(dict)  # Store rows grouped by index
    with open(path, 'r') as file:
        num_rows = int(file.readline().strip())  # Read the first line to get the number of rows
        for index, line in enumerate(file):
            row = line.strip().split()  # Split the line by whitespace
            type_key = row[0]  # The type of element ('P' or 'L')
            characteristics = set(row[2:])  # Store characteristics as a set
            data[index] = {"type": type_key, "characteristics": characteristics}
    return num_rows, data

# Highly accurate pairing for small datasets
def create_frames_accurate(data_dict):
    frames = {}
    frame_index = 0
    used_keys = set()
    p_elements = {key: value for key, value in data_dict.items() if value['type'] == 'P'}
    l_elements = {key: value for key, value in data_dict.items() if value['type'] == 'L'}

    # Exhaustive search for minimal overlap pairing
    while p_elements:
        key, row = p_elements.popitem()
        if key in used_keys:
            continue
        best_pair = None
        min_common_chars = float('inf')
        for other_key, other_row in p_elements.items():
            if other_key in used_keys:
                continue
            common_chars = len(row['characteristics'] & other_row['characteristics'])
            if common_chars < min_common_chars:
                min_common_chars = common_chars
                best_pair = other_key
        if best_pair is not None:
            frames[frame_index] = [key, best_pair]
            used_keys.update([key, best_pair])
            p_elements.pop(best_pair)
        else:
            frames[frame_index] = [key]
            used_keys.add(key)
        frame_index += 1

    # Add 'L' elements as single frames
    for key in l_elements:
        frames[frame_index] = [key]
        frame_index += 1

    return frames

# Performance-focused pairing for large datasets
def create_frames_fast(data_dict):
    frames = {}
    frame_index = 0
    used_keys = set()
    p_elements = {key: value for key, value in data_dict.items() if value['type'] == 'P'}
    l_elements = {key: value for key, value in data_dict.items() if value['type'] == 'L'}

    # Greedy approach with reduced comparisons
    while p_elements:
        key, row = p_elements.popitem()
        if key in used_keys:
            continue
        best_pair = None
        min_common_chars = float('inf')
        for other_key in list(p_elements.keys()):
            if other_key in used_keys:
                continue
            common_chars = len(row['characteristics'] & p_elements[other_key]['characteristics'])
            if common_chars < min_common_chars:
                min_common_chars = common_chars
                best_pair = other_key
        if best_pair is not None:
            frames[frame_index] = [key, best_pair]
            used_keys.update([key, best_pair])
            p_elements.pop(best_pair)
        else:
            frames[frame_index] = [key]
            used_keys.add(key)
        frame_index += 1

    # Add 'L' elements as single frames
    for key in l_elements:
        frames[frame_index] = [key]
        frame_index += 1

    return frames

# Scoring function
def calculate_score(frames, data_dict):
    score = 0
    frame_indices = list(frames.keys())
    
    for i in range(len(frame_indices) - 1):
        frame1 = frames[frame_indices[i]]
        frame2 = frames[frame_indices[i + 1]]
        
        tags1 = combine_tags(frame1, data_dict)
        tags2 = combine_tags(frame2, data_dict)
        
        local_score = min_form_number_of_common_tags(tags1, tags2)
        score += local_score
    
    return score

def combine_tags(frame, data_dict):
    combined_tags = set()
    for element in frame:
        combined_tags.update(data_dict[element]["characteristics"])
    return combined_tags

def min_form_number_of_common_tags(tags1, tags2):
    tags_in_tags1_not_in_tags2 = len(tags1 - tags2)
    tags_in_tags2_not_in_tags1 = len(tags2 - tags1)
    common_tags = len(tags1 & tags2)
    return min(tags_in_tags1_not_in_tags2, tags_in_tags2_not_in_tags1, common_tags)

# Reordering method (adjust based on scale if needed)
def reorder_frames(frames, data_dict):
    def get_frame_characteristics(frame):
        characteristics = set()
        for key in frame:
            characteristics.update(data_dict[key]["characteristics"])
        return characteristics

    frame_characteristics = {
        frame_index: get_frame_characteristics(frame)
        for frame_index, frame in frames.items()
    }

    frame_indices = list(frames.keys())
    ordered_frames = [frame_indices.pop(0)]
    remaining_frames = set(frame_indices)

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

    return {index: frames[frame] for index, frame in enumerate(ordered_frames)}

def write_frames_to_file(frames, output_path):
    with open(output_path, 'w') as file:
        file.write(f"{len(frames)}\n")  # Write the total number of frames
        for frame_index, frame in frames.items():
            frame_keys = " ".join(map(str, frame))  # Convert the frame elements to a space-separated string
            file.write(frame_keys + "\n")  # Write each frame

# Main execution
def process_file(path, output_path):
    num_rows, data_dict = load_txt_to_dict(path)

    if num_rows < 1000:
        print("Using highly accurate method for small dataset.")
        frames = create_frames_accurate(data_dict)
    elif num_rows < 10000:
        print("Using balanced method for medium dataset.")
        frames = create_frames_fast(data_dict)  # Current optimized code works well here
    else:
        print("Using performance-focused method for large dataset.")
        frames = create_frames_fast(data_dict)  # Fast method prioritizing performance

    reordered_frames = reorder_frames(frames, data_dict)
    score = calculate_score(reordered_frames, data_dict)
    print("Global Robotic Satisfaction Score:", score)
    write_frames_to_file(reordered_frames, output_path)

# Example usage
file_path1 = '../data/1_binary_landscapes.txt' #80 000
file_path2 = '../data/10_computable_moments.txt' #1 000
file_path3 = '../data/11_randomizing_paintings.txt' #90 000
file_path4 = '../data/110_oily_portraits.txt' #80 000




output_path = '../data/output_frames.txt'  # Path to the output file


file_path = '../data/example_dataset.txt'
output_path = '../data/output_frames.txt'
start = time.time()
process_file(file_path2, output_path)
end = time.time()
print("Execution time is:", end - start)
