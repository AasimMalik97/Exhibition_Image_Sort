import os
import heapq

def calculate_local_satisfaction(f1, f2):
    """Calculate the Local Robotic Satisfaction between two frameglasses."""
    common = len(f1 & f2)
    f1_diff = len(f1 - f2)
    f2_diff = len(f2 - f1)
    return min(common, f1_diff, f2_diff)

def pair_portraits_optimized(portraits):
    """Pair portraits efficiently using a heap to maximize diversity."""
    portrait_frameglasses = []
    heap = []

    # Precompute all possible pair diversities and store in a heap
    for i in range(len(portraits)):
        for j in range(i + 1, len(portraits)):
            diversity = len(portraits[i][1] | portraits[j][1])  # Union of tags
            heapq.heappush(heap, (-diversity, i, j))  # Max heap using negative diversity

    paired = set()
    while heap and len(paired) < len(portraits):
        _, i, j = heapq.heappop(heap)
        if i not in paired and j not in paired:
            combined_indices = portraits[i][0] + portraits[j][0]
            combined_tags = portraits[i][1] | portraits[j][1]
            portrait_frameglasses.append((combined_indices, combined_tags))
            paired.add(i)
            paired.add(j)

    # Add any leftover single portraits as frameglasses
    for idx in range(len(portraits)):
        if idx not in paired:
            portrait_frameglasses.append(portraits[idx])

    return portrait_frameglasses

def greedy_ordering_optimized(frameglasses):
    """Order frameglasses greedily with reduced complexity."""
    ordered = [frameglasses.pop(0)]  # Start with the first frameglass
    total_score = 0

    while frameglasses:
        # Precompute similarities with the current frameglass
        similarities = [
            (calculate_local_satisfaction(ordered[-1][1], f[1]), idx, f)
            for idx, f in enumerate(frameglasses)
        ]

        # Pick the best next frameglass
        best_score, best_index, best_frameglass = max(similarities)
        ordered.append(best_frameglass)
        total_score += best_score
        frameglasses.pop(best_index)

    return ordered, total_score

def process_paintings(file_path):
    """Read and process paintings from the input file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    paintings = []
    for line in lines[1:]:
        parts = line.strip().split()
        paintings.append([parts[0], int(parts[1]), *parts[2:]])

    landscapes = []
    portraits = []

    for idx, painting in enumerate(paintings):
        if painting[0] == 'L':
            landscapes.append(([idx], set(painting[2:])))
        elif painting[0] == 'P':
            portraits.append(([idx], set(painting[2:])))

    # Pair portraits using the optimized method
    portrait_frameglasses = pair_portraits_optimized(portraits)
    landscape_frameglasses = landscapes

    return portrait_frameglasses + landscape_frameglasses

def process_and_output(file_path, output_file_path):
    """Process input and write the output to a file."""
    frameglasses = process_paintings(file_path)
    ordered_frameglasses, max_score = greedy_ordering_optimized(frameglasses)
    num_frameglasses = len(ordered_frameglasses)
    
    # Write output to file
    with open(output_file_path, 'w') as f:
        f.write(f"{max_score}\n")  # Write the total score
        f.write(f"{num_frameglasses}\n")  # Write the number of frameglasses
        for frameglass in ordered_frameglasses:
            f.write(' '.join(map(str, frameglass[0])) + '\n')  # Write indices of the paintings

# Main Function
def main():
    # Define input and output file paths
    input_files = [
       "./Data/11_randomizing_paintings.txt",
    ]
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".txt", "_output.txt"))
        process_and_output(input_file, output_file)
        print(f"Processed {input_file}, output saved to {output_file}")

if __name__ == "__main__":
    main()
