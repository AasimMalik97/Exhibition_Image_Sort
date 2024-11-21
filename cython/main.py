from optimized_code import create_frames_fast_heuristic, reorder_frames_greedy, calculate_score, create_frames_fast_balanced, reorder_frames_greedy_lookahead
from datetime import datetime
import time

from memory_profiler import profile

def load_txt_to_dict(path):
    from collections import defaultdict
    data = defaultdict(dict)
    with open(path, 'r') as file:
        num_rows = int(file.readline().strip())
        for index, line in enumerate(file):
            row = line.strip().split()
            type_key = row[0]
            characteristics = set(row[2:])
            data[index] = {"type": type_key, "characteristics": characteristics}
    return num_rows, dict(data)

def write_frames_to_file(frames, output_path):
    with open(output_path, 'w') as file:
        file.write(f"{len(frames)}\n")
        for frame_index, frame in frames.items():
            file.write(" ".join(map(str, frame)) + "\n")


@profile
def process_file(input_path, output_path):
    num_rows, data_dict = load_txt_to_dict(input_path)
    frames = create_frames_fast_balanced(data_dict)
    reordered_frames = reorder_frames_greedy_lookahead(frames, data_dict)
    score = calculate_score(reordered_frames, data_dict)
    print(f"Global Robotic Satisfaction Score: {score}")
    write_frames_to_file(reordered_frames, output_path)


if __name__ == "__main__":
    file_path1 = '../../data/1_binary_landscapes.txt' #80 000
    file_path2 = '../../data/10_computable_moments.txt'
    file_path10 = '../../data/15000.txt' #15 000
    file_path3 = '../../data/11_randomizing_paintings.txt' #90 000
    file_path4 = '../../data/110_oily_portraits.txt' #80 000
    input_path = "example_dataset.txt"
    output_path = "output_frames.txt"
    start = time.time()
    currentTime = datetime.now()
    print("Start Time:", currentTime.strftime("%H:%M:%S"))
    
    process_file(file_path4, output_path)
    end = time.time()
    #lookahead for 14min is: 16150
    print(end - start)



