import json
import time
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Iterator
from heapq import heappush, heappop, heapreplace, nlargest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from dataclasses import dataclass
from itertools import islice
import gc

@dataclass
class ImageData:
    id: List[int]
    type: str
    tags: Set[str]
    tag_length: int

class LargeScaleProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.portrait_images: List[ImageData] = []
        self.landscape_images: List[ImageData] = []
        self.frame_portrait_images: List[ImageData] = []
        self.tag_to_images: Dict[str, Set[int]] = defaultdict(set)
        self.image_to_tags: Dict[int, Set[str]] = {}
        self.num_cores = max(1, multiprocessing.cpu_count() - 1)
        self.chunk_size = 1000  # Reduced chunk size for better memory management
        self.pair_batch_size = 10000  # Number of pairs to process at once
        self.top_pairs_to_keep = 10000  # Number of best pairs to keep in memory

    def _process_line(self, line_data: Tuple[int, str]) -> Tuple[ImageData, int]:
        """Process a single line of input data"""
        idx, line = line_data
        if not line.strip():
            return None
        
        parts = line.split()
        return ImageData(
            id=[idx],
            type=parts[0],
            tags=set(parts[2:]),
            tag_length=int(parts[1])
        ), idx

    def parse_input_file(self) -> None:
        """Enhanced parallel file parsing with memory efficiency"""
        with open(self.file_path, 'r') as file:
            next(file)  # Skip header
            lines = [(idx, line) for idx, line in enumerate(file)]

        # Process in manageable chunks
        chunks = [lines[i:i + self.chunk_size] for i in range(0, len(lines), self.chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            for chunk_results in executor.map(self._process_chunk, chunks):
                for image_data, idx in chunk_results:
                    if image_data:
                        if image_data.type == "P":
                            self.portrait_images.append(image_data)
                        else:
                            self.landscape_images.append(image_data)

    def _process_chunk(self, chunk: List[Tuple[int, str]]) -> List[Tuple[ImageData, int]]:
        """Process a chunk of lines in parallel"""
        return [self._process_line(line_data) for line_data in chunk if line_data[1].strip()]

    def _generate_pairs_iterator(self, n: int) -> Iterator[Tuple[int, int]]:
        """Generate pairs of indices in a memory-efficient way"""
        for i in range(n):
            # Generate pairs for current i in chunks
            for j in range(i + 1, n, self.chunk_size):
                end = min(j + self.chunk_size, n)
                yield from ((i, k) for k in range(j, end))

    def _process_pair_batch(self, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        """Process a batch of pairs and return their similarities"""
        results = []
        for i, j in pairs:
            img1, img2 = self.portrait_images[i], self.portrait_images[j]
            intersection = len(img1.tags & img2.tags)
            similarity = min(
                len(img1.tags) - intersection,
                len(img2.tags) - intersection,
                intersection
            )
            results.append((similarity, len(img1.tags | img2.tags), i, j))
        return results

    def find_optimal_portrait_pairs(self) -> None:
        """
        Efficiently pair portrait images with the least number of common tags
        """
        if len(self.portrait_images) < 2:
            return

        Frame_Portrait_Images = []
        Temp_Portrait_Images = self.portrait_images.copy()

        while Temp_Portrait_Images:
            element1 = Temp_Portrait_Images.pop(0)
            
            for j in range(len(Temp_Portrait_Images)):
                element2 = Temp_Portrait_Images[j]
                common_tags = len(set(element1.tags) & set(element2.tags))
                
                combined_tags = list(set(element1.tags) | set(element2.tags))
                if common_tags < 1 or j == len(Temp_Portrait_Images) - 1:
                    Frame_Portrait_Images.append(ImageData(
                        id=element1.id + element2.id,
                        type="P",
                        tags=set(combined_tags),
                        tag_length=len(combined_tags)
                    ))
                    Temp_Portrait_Images.pop(j)
                    break

        # Update the frame portrait images with the new pairs
        self.frame_portrait_images = Frame_Portrait_Images

        # Logging
        print(f"Paired Images: {len(Frame_Portrait_Images)}")
        print(f"Original Images: {len(self.portrait_images)}")
    
    def build_tag_index(self, final_list: List[ImageData]) -> None:
        """Build optimized tag index with parallel processing"""
        self.tag_to_images.clear()
        self.image_to_tags.clear()
        
        chunks = [final_list[i:i + self.chunk_size] 
                 for i in range(0, len(final_list), self.chunk_size)]
        
        with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            futures = []
            for chunk_idx, chunk in enumerate(chunks):
                futures.append(executor.submit(
                    self._process_index_chunk, chunk, chunk_idx * self.chunk_size
                ))
            
            for future in futures:
                local_tag_to_images, local_image_to_tags = future.result()
                for tag, indices in local_tag_to_images.items():
                    self.tag_to_images[tag].update(indices)
                self.image_to_tags.update(local_image_to_tags)

    def _process_index_chunk(self, chunk: List[ImageData], base_idx: int) -> Tuple[Dict[str, Set[int]], Dict[int, Set[str]]]:
        """Process a chunk of images for indexing"""
        local_tag_to_images = defaultdict(set)
        local_image_to_tags = {}
        
        for idx, image in enumerate(chunk, start=base_idx):
            for tag in image.tags:
                local_tag_to_images[tag].add(idx)
            local_image_to_tags[idx] = image.tags
            
        return local_tag_to_images, local_image_to_tags

    def sort_images_fast(self) -> List[ImageData]:
        """Optimized image sorting with caching and parallel processing"""
        final_list = self.frame_portrait_images + self.landscape_images
        if not final_list:
            return []

        self.build_tag_index(final_list)
        
        sorted_images = [final_list[0]]
        used_indices = {0}
        similarity_cache = {}

        while len(sorted_images) < len(final_list):
            print(f"Images sorted: {len(sorted_images)} / {len(final_list)}", end='\r')
            
            current_idx = len(sorted_images) - 1
            current_tags = sorted_images[-1].tags

            # Get candidate indices efficiently
            candidate_indices = set()
            for tag in current_tags:
                candidate_indices.update(self.tag_to_images[tag])
            candidate_indices -= used_indices

            if not candidate_indices:
                # Take first unused image if no candidates found
                for i in range(len(final_list)):
                    if i not in used_indices:
                        sorted_images.append(final_list[i])
                        used_indices.add(i)
                        break
                continue

            # Calculate similarities in parallel for new candidates
            chunks = [list(candidate_indices)[i:i + self.chunk_size] 
                     for i in range(0, len(candidate_indices), self.chunk_size)]

            best_similarity = float('-inf')
            best_idx = None

            with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
                futures = []
                for chunk in chunks:
                    futures.append(executor.submit(
                        self._find_best_candidate,
                        chunk,
                        current_tags,
                        final_list,
                        similarity_cache
                    ))
                
                for future in futures:
                    sim, idx = future.result()
                    if sim > best_similarity:
                        best_similarity = sim
                        best_idx = idx

            if best_idx is not None:
                sorted_images.append(final_list[best_idx])
                used_indices.add(best_idx)

        return sorted_images

    def _find_best_candidate(self, chunk: List[int], current_tags: Set[str], 
                           final_list: List[ImageData], cache: Dict) -> Tuple[int, int]:
        """Find the best candidate from a chunk"""
        best_similarity = float('-inf')
        best_idx = None
        
        for idx in chunk:
            cache_key = (frozenset(current_tags), frozenset(final_list[idx].tags))
            if cache_key not in cache:
                cache[cache_key] = len(current_tags & final_list[idx].tags)
            
            similarity = cache[cache_key]
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
                
        return best_similarity, best_idx

    def calculate_final_score(self, sorted_images: List[ImageData]) -> Dict[str, float]:
        """Calculate the final score of the image sorting"""
        if not sorted_images or len(sorted_images) < 2:
            return {
                "total_score": 0,
                "average_score": 0,
                "total_transitions": 0,
                "min_similarity": 0,
                "max_similarity": 0
            }

        total_score = 0
        min_similarity = float('inf')
        max_similarity = float('-inf')
        
        for i in range(len(sorted_images) - 1):
            similarity = len(sorted_images[i].tags & sorted_images[i + 1].tags)
            total_score += similarity
            min_similarity = min(min_similarity, similarity)
            max_similarity = max(max_similarity, similarity)
        
        num_transitions = len(sorted_images) - 1
        average_score = total_score / num_transitions if num_transitions > 0 else 0
        
        return {
            "total_score": total_score,
            "average_score": round(average_score, 2),
            "total_transitions": num_transitions,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity
        }

    def write_output(self, sorted_images: List[ImageData], output_path: str) -> None:
        """Write output with buffering for better performance and include scoring information"""
        # Calculate scores
        scores = self.calculate_final_score(sorted_images)
        
        with open(output_path, 'w', buffering=8192) as file:
            # Write the original output
            file.write(f"{len(sorted_images)}\n")
            for image in sorted_images:
                file.write(f"{' '.join(map(str, image.id))}\n")
            
def main():
    start_time = time.time()
    
    processor = LargeScaleProcessor('../Data/110_oily_portraits.txt')  # Replace with your input file path
    print("Parsing input file...")
    processor.parse_input_file()
    
    print("Finding optimal portrait pairs...")
    processor.find_optimal_portrait_pairs()
    
    print("Sorting images...")
    sorted_images = processor.sort_images_fast()
    
    # Calculate and display scores
    scores = processor.calculate_final_score(sorted_images)
    print("\nSorting Statistics:")
    print(f"Total Score: {scores['total_score']}")
    print(f"Average Score per Transition: {scores['average_score']}")
    print(f"Total Transitions: {scores['total_transitions']}")
    print(f"Minimum Similarity: {scores['min_similarity']}")
    print(f"Maximum Similarity: {scores['max_similarity']}")
    
    print("\nWriting output...")
    processor.write_output(sorted_images, "../Data/output110.txt")  # Replace with your output file path
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
    
