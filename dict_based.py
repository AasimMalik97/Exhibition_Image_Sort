from collections import defaultdict
import time


# Function to load data from a text file into a nested dictionary
def load_txt_to_dict(path):
    data = defaultdict(dict)  # Use defaultdict to store rows grouped by index
    with open(path, 'r') as file:
        num_rows = int(file.readline().strip())  # Read the number of rows (not used here)
        for index, line in enumerate(file):
            row = line.strip().split()  # Split the line by whitespace
            print(index)
            type_key = row[0]  # The type of element ('P' or 'L')
            num_characteristics = int(row[1])  # Number of characteristics
            characteristics = set(row[2:])  # Store characteristics as a set for efficient operations

            # Store each row's data in the dictionary
            data[index] = {
                "type": type_key,
                "num_characteristics": num_characteristics,
                "characteristics": characteristics
            }

    return data












#0.3 sec for 1000
# Function to create frames from the data
def create_frames1(data_dict):
    frames = {}  # Dictionary to store frames
    frame_index = 0  # Frame counter
    used_keys = set()  # Set to track already used keys

    # Separate elements by type ('P' and 'L')
    p_elements = {key: value for key, value in data_dict.items() if value['type'] == 'P'}
    l_elements = {key: value for key, value in data_dict.items() if value['type'] == 'L'}

    # Handle 'P' type frames
    while p_elements:
        key, row = p_elements.popitem()  # Take an arbitrary 'P' element
        if key in used_keys:  # Skip if already used
            continue

        # Find the best pair with the least common characteristics
        best_pair = None
        min_common_chars = float('inf')  # Initialize with a high value

        for other_key, other_row in p_elements.items():
            if other_key in used_keys:  # Skip already used keys
                continue
            common_chars = len(row['characteristics'] & other_row['characteristics'])  # Count common characteristics
            if common_chars < min_common_chars:  # Update if a better pair is found
                min_common_chars = common_chars
                best_pair = other_key
        print(best_pair)
        if best_pair is not None:
            # Pair the current 'P' element with the best match
            frames[frame_index] = [key, best_pair]
            used_keys.update([key, best_pair])  # Mark both keys as used
            del p_elements[best_pair]  # Remove the best pair from available 'P' elements
        else:
            # No pair found, create a single frame
            frames[frame_index] = [key]
            used_keys.add(key)  # Mark the key as used

        frame_index += 1  # Increment the frame index

    # Handle 'L' type frames (each is its own frame)
    for key in l_elements.keys():
        if key not in used_keys:  # Only process unused keys
            frames[frame_index] = [key]  # Add as a single frame
            used_keys.add(key)  # Mark the key as used
            frame_index += 1  # Increment the frame index

    return frames

start2= time.time()

#27 sec
import heapq
def create_frames(data_dict):
    frames = {}
    frame_index = 0
    used_keys = set()

    # Separate 'P' and 'L' types
    p_elements = {key: value for key, value in data_dict.items() if value['type'] == 'P'}
    l_elements = {key: value for key, value in data_dict.items() if value['type'] == 'L'}

    # Build a priority queue of (common characteristics, key1, key2)
    pq = []
    keys = list(p_elements.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1, key2 = keys[i], keys[j]
            common_chars = len(p_elements[key1]['characteristics'] & p_elements[key2]['characteristics'])
            heapq.heappush(pq, (common_chars, key1, key2))
            

    # Pair 'P' elements using the priority queue
    while pq:
        _, key1, key2 = heapq.heappop(pq)
        if key1 not in used_keys and key2 not in used_keys:
            frames[frame_index] = [key1, key2]
            used_keys.update([key1, key2])
            frame_index += 1

    # Add remaining single 'P' elements as individual frames
    for key in p_elements.keys():
        if key not in used_keys:
            frames[frame_index] = [key]
            used_keys.add(key)
            frame_index += 1

    # Add 'L' type elements as individual frames
    for key in l_elements.keys():
        frames[frame_index] = [key]
        frame_index += 1

    return frames







# Function to reorder frames based on the most common characteristics
def reorder_frames1(frames, data_dict):
    # Helper function to get all characteristics from a frame
    def get_frame_characteristics(frame):
        characteristics = set()
        for key in frame:  # For each element in the frame
            characteristics.update(data_dict[key]["characteristics"])  # Add its characteristics
        return characteristics

    # Start with an arbitrary frame as the first frame
    frame_keys = list(frames.keys())  # Get all frame indices
    current_frame_key = frame_keys.pop(0)  # Start with the first frame
    ordered_frames = [current_frame_key]  # Initialize the ordered frame list
    remaining_frames = set(frame_keys)  # Track remaining frames

    # Greedy algorithm to reorder frames
    while remaining_frames:
        current_characters = get_frame_characteristics(frames[current_frame_key])  # Get characteristics of the current frame
        max_common = -1  # Initialize with a low value
        best_next_frame = None  # Initialize the best next frame

        for next_frame_key in remaining_frames:  # Check all remaining frames
            next_characters = get_frame_characteristics(frames[next_frame_key])  # Get characteristics of the candidate frame
            common_characters = len(current_characters & next_characters)  # Count common characteristics
            if common_characters > max_common:  # Update if a better match is found
                max_common = common_characters
                best_next_frame = next_frame_key

        if best_next_frame is not None:
            # Add the best next frame to the ordered list
            ordered_frames.append(best_next_frame)
            remaining_frames.remove(best_next_frame)  # Remove it from the remaining frames
            current_frame_key = best_next_frame  # Update the current frame

    # Reconstruct the frames dictionary in the new order
    reordered_frames = {index: frames[frame_key] for index, frame_key in enumerate(ordered_frames)}
    return reordered_frames


#27 sec for 1000
#76 sec for 1416
#172 sec for 2000

def reorder_frames(frames, data_dict):
    # Helper to calculate similarity between frames
    def calculate_similarity(frame1, frame2):
        chars1 = get_frame_characteristics(frame1)
        chars2 = get_frame_characteristics(frame2)
        return len(chars1 & chars2)  # Number of common characteristics

    # Helper to get all characteristics from a frame
    def get_frame_characteristics(frame):
        characteristics = set()
        for key in frame:
            characteristics.update(data_dict[key]["characteristics"])
        return characteristics

    # Create similarity matrix
    frame_keys = list(frames.keys())
    similarity_matrix = {}
    for i in range(len(frame_keys)):
        for j in range(i + 1, len(frame_keys)):
            similarity = calculate_similarity(frames[frame_keys[i]], frames[frame_keys[j]])
            similarity_matrix[(frame_keys[i], frame_keys[j])] = similarity
            similarity_matrix[(frame_keys[j], frame_keys[i])] = similarity

    # Use a greedy MST algorithm (Prim's)
    ordered_frames = [frame_keys[0]]
    remaining_frames = set(frame_keys[1:])
    while remaining_frames:
        best_next_frame = None
        max_similarity = -1
        for current_frame in ordered_frames:
            for candidate_frame in remaining_frames:
                similarity = similarity_matrix[(current_frame, candidate_frame)]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_next_frame = candidate_frame

        ordered_frames.append(best_next_frame)
        remaining_frames.remove(best_next_frame)

    # Reconstruct frames in the new order
    return {index: frames[frame_key] for index, frame_key in enumerate(ordered_frames)}


# Function to write frames to an output file
def write_frames_to_file(frames, output_path):
    with open(output_path, 'w') as file:
        file.write(f"{len(frames)}\n")  # Write the number of frames
        for frame_index, frame in frames.items():  # Iterate over frames
            frame_keys = " ".join(map(str, frame))  # Convert frame elements to strings
            file.write(frame_keys +"\n")  # Write each frame

# Example usage
file_path1 = '../data/1_binary_landscapes.txt'
file_path2 = '../data/10_computable_moments.txt'
file_path3 = '../data/11_randomizing_paintings.txt'
file_path4 = '../data/110_oily_portraits.txt'
file_path5 = '../data/11_30000.txt'
file_path6 = '../data/1000.txt' #1000
file_path7 = '../data/2000.txt' #2000
file_path8 = '../data/4000.txt' #4000


output_path = '../data/output_frames.txt'  # Path to the output file


# Load data from the input file
data_dict = load_txt_to_dict(file_path8)

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