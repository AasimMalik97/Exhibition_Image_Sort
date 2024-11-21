import os

def calculate_local_satisfaction(f1, f2):
    """Calculate the Local Robotic Satisfaction between two frameglasses."""
    common = len(f1 & f2)
    f1_diff = len(f1 - f2)
    f2_diff = len(f2 - f1)
    return min(common, f1_diff, f2_diff)

def pair_portraits_optimized(portraits):
    """Pair portraits greedily with reduced complexity."""
    portrait_frameglasses = []

    # Sort portraits by tag set size (heuristic for diverse pairing)
    portraits = sorted(portraits, key=lambda x: len(x[1]))

    while len(portraits) > 1:
        # Always pick the smallest portrait (most unique tags)
        p1 = portraits.pop(0)

        # Find the portrait that maximizes diversity with p1
        best_match = None
        best_diversity = -1
        best_index = -1

        for idx, p2 in enumerate(portraits):
            diversity = len(p1[1] | p2[1])  # Union of tags
            if diversity > best_diversity:
                best_diversity = diversity
                best_match = p2
                best_index = idx

        # Combine p1 and best_match into a frameglass
        combined_indices = p1[0] + best_match[0]
        combined_tags = p1[1] | best_match[1]
        portrait_frameglasses.append((combined_indices, combined_tags))

        # Remove the matched portrait
        portraits.pop(best_index)

    # Add any leftover single portrait as a frameglass
    if portraits:
        portrait_frameglasses.append(portraits.pop())

    return portrait_frameglasses

def greedy_ordering_optimized(frameglasses):
    """Order frameglasses greedily with reduced complexity."""
    ordered = [frameglasses.pop(0)]  # Start with the first frameglass
    total_score = 0

    while frameglasses:
        best_next = None
        best_score = -1
        best_index = -1

        # Precompute similarities only with the last added frameglass
        for idx, f in enumerate(frameglasses):
            score = calculate_local_satisfaction(ordered[-1][1], f[1])
            if score > best_score:
                best_score = score
                best_next = f
                best_index = idx

        # Add the best next frameglass to the ordered list
        ordered.append(best_next)
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
        f.write(f"{max_score}\n")
        f.write(f"{num_frameglasses}\n")  # Write the number of frameglasses
        for frameglass in ordered_frameglasses:
            f.write(' '.join(map(str, frameglass[0])) + '\n')  # Write indices of the paintings

# Main Function
def main():
    # Define input and output file paths
    input_files = [
      #  "./Data/0_example.txt",
        #"./Data/10_computable_moments.txt",
        "./Data/11_randomizing_paintings.txt",
     #   "110_oily_portraits.txt"
    ]
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".txt", "_output.txt"))
        process_and_output(input_file, output_file)
        print(f"Processed {input_file}, output saved to {output_file}")

if __name__ == "__main__":
    main()
