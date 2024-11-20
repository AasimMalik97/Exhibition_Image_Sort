import os
import heapq

def process_large_file_with_quick_heap(file_path, output_path):
    """Processes a large input file with quicksort and heap-based optimization."""
    batch_size = 5000
    frameglasses = []
    total_score = 0

    def parse_line(line):
        parts = line.strip().split()
        orientation = parts[0]
        tags = set(parts[2:])
        return {"orientation": orientation, "tags": tags}

    def calculate_pair_score(tags1, tags2):
        common_tags = len(tags1 & tags2)
        tags_in_f1_not_in_f2 = len(tags1 - tags2)
        tags_in_f2_not_in_f1 = len(tags2 - tags1)
        return min(common_tags, tags_in_f1_not_in_f2, tags_in_f2_not_in_f1)

    def calculate_batch_score(batch):
        score = 0
        for i in range(len(batch) - 1):
            score += calculate_pair_score(batch[i]["tags"], batch[i + 1]["tags"])
        return score

    def quicksort(batch, reference_tags):
        stack = [(batch, reference_tags)]
        sorted_result = []

        while stack:
            current_batch, ref_tags = stack.pop()
            if len(current_batch) <= 1:
                sorted_result.extend(current_batch)
                continue
            
            pivot = current_batch[0]
            pivot_score = calculate_pair_score(ref_tags, pivot["tags"])

            less = [fg for fg in current_batch[1:] if calculate_pair_score(ref_tags, fg["tags"]) <= pivot_score]
            greater = [fg for fg in current_batch[1:] if calculate_pair_score(ref_tags, fg["tags"]) > pivot_score]

            if greater:
                stack.append((greater, ref_tags))
            sorted_result.append(pivot)
            if less:
                stack.append((less, ref_tags))

        return sorted_result[::-1]

    def heap_based_sort(batch):
        """Refines the order of frameglasses using a max heap."""
        heap = []
        sorted_batch = [batch.pop(0)]  # Start with the first frameglass

        # Build initial heap with unique identifiers
        for idx, fg in enumerate(batch):
            score = calculate_pair_score(sorted_batch[-1]["tags"], fg["tags"])
            heapq.heappush(heap, (-score, idx, fg))  # Use idx as a tie-breaker

        while heap:
            # Pop the best frameglass based on the highest score
            _, _, best_fg = heapq.heappop(heap)
            sorted_batch.append(best_fg)

            # Update heap with new scores relative to the last added frameglass
            new_heap = []
            for _, idx, fg in heap:
                new_score = calculate_pair_score(sorted_batch[-1]["tags"], fg["tags"])
                heapq.heappush(new_heap, (-new_score, idx, fg))
            heap = new_heap

        return sorted_batch

    with open(file_path, 'r') as file:
        batch = []
        for line in file:
            if len(batch) < batch_size:
                batch.append(parse_line(line))
            else:
                batch = quicksort(batch, batch[0]["tags"])
                batch = heap_based_sort(batch)
                frameglasses.extend(batch)
                total_score += calculate_batch_score(batch)
                batch = []

        if batch:
            batch = quicksort(batch, batch[0]["tags"])
            batch = heap_based_sort(batch)
            frameglasses.extend(batch)
            total_score += calculate_batch_score(batch)

    with open(output_path, 'w') as out_file:
        out_file.write(f"Total Score: {total_score}\n")
        for fg in frameglasses:
            out_file.write(f"{fg['orientation']} {len(fg['tags'])} {' '.join(fg['tags'])}\n")

    print(f"Processing completed. Total Score: {total_score}")

input_file = "./Data/10_computable_moments.txt"
output_file = "quick_heap_output_with_fix_10.txt"
process_large_file_with_quick_heap(input_file, output_file)
