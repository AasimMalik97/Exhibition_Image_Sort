import os

def calculate_local_satisfaction(f1, f2):
    """Calculate the Local Robotic Satisfaction between two frameglasses."""
    common = len(f1 & f2)
    f1_diff = len(f1 - f2)
    f2_diff = len(f2 - f1)
    return min(common, f1_diff, f2_diff)

def pair_portraits_greedy(portraits):
    """Greedy strategy to pair portraits for maximizing diversity."""
    paired = []
    while len(portraits) > 1:
        p1 = portraits.pop(0)
        best_match = None
        best_diversity = -1

        for idx, p2 in enumerate(portraits):
            diversity = len(p1[1] | p2[1])  # Union of tags
            if diversity > best_diversity:
                best_diversity = diversity
                best_match = (idx, p2)

        if best_match:
            idx, p2 = best_match
            combined_indices = p1[0] + p2[0]
            combined_tags = p1[1] | p2[1]
            paired.append((combined_indices, combined_tags))
            portraits.pop(idx)

    # Add leftover portraits as single frameglasses
    paired.extend(portraits)
    return paired

def greedy_ordering_with_refinement(frameglasses):
    """Greedy ordering with local refinement for better scores."""
    ordered = [frameglasses.pop(0)]
    total_score = 0

    while frameglasses:
        similarities = [
            (calculate_local_satisfaction(ordered[-1][1], f[1]), idx, f)
            for idx, f in enumerate(frameglasses)
        ]
        best_score, best_index, best_frameglass = max(similarities)
        ordered.append(best_frameglass)
        total_score += best_score
        frameglasses.pop(best_index)

    # Local refinement: Swap neighboring frameglasses for better scores
    improved = True
    while improved:
        improved = False
        for i in range(1, len(ordered) - 1):
            current_score = calculate_local_satisfaction(ordered[i - 1][1], ordered[i][1]) + \
                            calculate_local_satisfaction(ordered[i][1], ordered[i + 1][1])
            swapped_score = calculate_local_satisfaction(ordered[i - 1][1], ordered[i + 1][1]) + \
                            calculate_local_satisfaction(ordered[i + 1][1], ordered[i][1])
            if swapped_score > current_score:
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]
                total_score += swapped_score - current_score
                improved = True

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

    # Pair portraits using greedy strategy
    portrait_frameglasses = pair_portraits_greedy(portraits)
    landscape_frameglasses = landscapes

    return portrait_frameglasses + landscape_frameglasses

def process_and_output(file_path, output_file_path):
    """Process input and write the output to a file."""
    frameglasses = process_paintings(file_path)
    ordered_frameglasses, max_score = greedy_ordering_with_refinement(frameglasses)
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
        #"0_example.txt",
        #"./Data/10_computable_moments.txt",
      #"./Data/11_randomizing_paintings.txt",
      #  "110_oily_portraits.txt"
      "./Data/1_binary_landscapes.txt",
    ]
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".txt", "_output.txt"))
        process_and_output(input_file, output_file)
        print(f"Processed {input_file}, output saved to {output_file}")

if __name__ == "__main__":
    main()
