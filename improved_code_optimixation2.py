import json
import time
import networkx as nx
import heapq
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Iterator
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from itertools import combinations
import concurrent.futures
import math

@dataclass
class ImageData:
    id: List[int]
    type: str
    tags: Set[str]
    tag_length: int

    def merge_with(self, other: 'ImageData') -> 'ImageData':
        """Merge two ImageData objects"""
        return ImageData(
            id=self.id + other.id,
            type=self.type,
            tags=self.tags.union(other.tags),
            tag_length=len(self.tags.union(other.tags))
        )

class LargeScaleProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.portrait_images: List[ImageData] = []
        self.landscape_images: List[ImageData] = []
        self.frame_portrait_images: List[ImageData] = []
        self.tag_to_images: Dict[str, Set[int]] = defaultdict(set)
        self.image_to_tags: Dict[int, Set[str]] = {}
        self.tag_frequencies: Dict[str, int] = defaultdict(int)
        self.tag_weights: Dict[str, float] = {}
        self.chunk_size = 5000  # Increased chunk size for better performance
        self.max_pairs_per_batch = 10000  # Maximum pairs to process in memory at once

    def parse_input_file(self) -> None:
        """Parse input file with optimized batch processing"""
        try:
            with open(self.file_path, 'r') as file:
                # Read header
                header = next(file)
                total_images = int(header.strip())
                
                # Initialize progress bar
                with tqdm(total=total_images, desc="Parsing file") as pbar:
                    buffer = []
                    for line in file:
                        if len(buffer) >= self.chunk_size:
                            self._process_batch(buffer)
                            pbar.update(len(buffer))
                            buffer = []
                        buffer.append(line)
                    
                    # Process remaining lines
                    if buffer:
                        self._process_batch(buffer)
                        pbar.update(len(buffer))
            
            # Calculate weights after processing all images
            self._calculate_tag_weights()
            
        except Exception as e:
            print(f"Error parsing file: {str(e)}")
            raise

    def _process_batch(self, lines: List[str]) -> None:
        """Process a batch of lines efficiently"""
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.split()
            if len(parts) < 3:
                continue
                
            image_type = parts[0]
            tag_length = int(parts[1])
            tags = set(parts[2:])
            
            # Update tag frequencies
            for tag in tags:
                self.tag_frequencies[tag] += 1
            
            # Create ImageData object
            image_data = ImageData(
                id=[len(self.portrait_images) + len(self.landscape_images)],
                type=image_type,
                tags=tags,
                tag_length=tag_length
            )
            
            # Add to appropriate list
            if image_type == "P":
                self.portrait_images.append(image_data)
            else:
                self.landscape_images.append(image_data)

    def _calculate_tag_weights(self) -> None:
        """Calculate optimized tag weights"""
        if not self.tag_frequencies:
            return
            
        # Use log-based IDF weights
        total_images = len(self.portrait_images) + len(self.landscape_images)
        self.tag_weights = {
            tag: math.log(total_images / (freq + 1)) + 1
            for tag, freq in self.tag_frequencies.items()
        }

    def find_optimal_portrait_pairs(self) -> None:
        """Greedily pair portrait images, each with the most similar next available image, ensuring no image is paired with itself."""
        if len(self.portrait_images) < 2:
            return

        print(f"Processing {len(self.portrait_images)} portrait images...")

        # Precompute candidates for each image based on shared tags
        self._build_indices(self.portrait_images)
        similarity_cache = {}

        used_indices = set()  # Track images already paired
        paired_images = []

        # Use a min-heap for pairing by least common similarity
        heap = []

        # Generate initial pairs for all unpaired images
        for i in range(len(self.portrait_images)):
            if i in used_indices:
                continue
            
            print(f"Processing image {i}...")

            img1 = self.portrait_images[i]
            # Generate candidates
            candidates = self._get_candidates(img1, set(range(len(self.portrait_images))) - used_indices)

            for j in candidates:
                if j in used_indices:
                    continue

                # Check if similarity score is already calculated
                cache_key = (i, j)
                if cache_key not in similarity_cache:
                    similarity = self.calculate_similarity(img1, self.portrait_images[j])
                    similarity_cache[cache_key] = similarity
                else:
                    similarity = similarity_cache[cache_key]

                if similarity > 0:
                    heapq.heappush(heap, (-similarity, i, j))  # Use the negative similarity for max-heap simulation

        start_time = time.time()
        while heap and len(used_indices) < len(self.portrait_images):
            # If the process takes more than 5 minutes, we should stop
            if time.time() - start_time > 300:
                print("Timeout reached. Stopping early.")
                break

            # Extract the most similar pair from the heap
            similarity, i, j = heapq.heappop(heap)

            # Ensure that we do not pair an image with itself and both images are unpaired
            if i != j and i not in used_indices and j not in used_indices:
                # Merge the best pair
                merged = self.portrait_images[i].merge_with(self.portrait_images[j])
                paired_images.append(merged)

                # Mark both images as used
                used_indices.update([i, j])

                # Remove processed pairs from the heap
                for k in list(heap):
                    if k[1] == i or k[2] == j:  # Remove pairs where either image is already used
                        heap.remove(k)
                heapq.heapify(heap)  # Rebalance heap after removal

        self.frame_portrait_images = paired_images
        print(f"Created {len(paired_images)} paired images (expected: {len(self.portrait_images) // 2})")
    
    def _find_batch_pairs(self, batch: List[ImageData], offset: int) -> List[Tuple[int, int, float]]:
        """Find optimal pairs within a batch"""
        pairs = []
        
        # Calculate similarities for all possible pairs in the batch
        for i, img1 in enumerate(batch):
            for j in range(i + 1, len(batch)):
                img2 = batch[j]
                similarity = self.calculate_similarity(img1, img2)
                if similarity > 0:
                    pairs.append((i + offset, j + offset, similarity))
        
        # Sort pairs by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def calculate_similarity(self, img1: ImageData, img2: ImageData) -> float:
        """Optimized similarity calculation"""
        common_tags = img1.tags.intersection(img2.tags)
        if not common_tags:
            return 0.0
        
        similarity = sum(self.tag_weights.get(tag, 1.0) for tag in common_tags)
        normalization = math.sqrt(len(img1.tags) * len(img2.tags))
        
        return similarity / normalization if normalization > 0 else 0.0

    def sort_images(self) -> List[ImageData]:
        """Memory-efficient sorting for large datasets"""
        final_list = self.frame_portrait_images + self.landscape_images
        if not final_list:
            return []

        # Build efficient index structures
        self._build_indices(final_list)
        
        sorted_indices = [0]  # Start with first image
        available_indices = set(range(1, len(final_list)))
        
        batch_size = 1000  # Process candidates in batches
        
        with tqdm(total=len(final_list)-1, desc="Sorting images") as pbar:
            while available_indices:
                current_idx = sorted_indices[-1]
                current_image = final_list[current_idx]
                
                # Get candidate indices through tag index
                candidates = self._get_candidates(current_image, available_indices)
                
                if candidates:
                    # Process candidates in batches
                    best_similarity = float('-inf')
                    best_idx = None
                    
                    for i in range(0, len(candidates), batch_size):
                        batch = list(candidates)[i:i + batch_size]
                        for idx in batch:
                            similarity = self.calculate_similarity(
                                current_image,
                                final_list[idx]
                            )
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_idx = idx
                    
                    if best_idx is not None:
                        sorted_indices.append(best_idx)
                        available_indices.remove(best_idx)
                else:
                    # If no candidates, take first available index
                    next_idx = min(available_indices)
                    sorted_indices.append(next_idx)
                    available_indices.remove(next_idx)
                
                pbar.update(1)
        
        return [final_list[i] for i in sorted_indices]

    def _get_candidates(self, image: ImageData, available_indices: Set[int]) -> Set[int]:
        """Get candidate indices efficiently"""
        candidates = set()
        for tag in image.tags:
            candidates.update(self.tag_to_images[tag])
        return candidates.intersection(available_indices)

    def _build_indices(self, images: List[ImageData]) -> None:
        """Build efficient tag indices"""
        self.tag_to_images.clear()
        self.image_to_tags.clear()
        
        for idx, image in enumerate(images):
            self.image_to_tags[idx] = image.tags
            for tag in image.tags:
                self.tag_to_images[tag].add(idx)

    def calculate_final_score(self, sorted_images: List[ImageData]) -> Dict[str, float]:
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
        
        print(f"Total score: {total_score}")

        return {
            "total_score": total_score,
            "average_score": round(average_score, 2),
            "total_transitions": num_transitions,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity
        }

    def write_output(self, sorted_images: List[ImageData], output_path: str) -> None:
        """Write output efficiently"""
        with open(output_path, 'w') as file:
            file.write(f"{len(sorted_images)}\n")
            for image in sorted_images:
                file.write(f"{' '.join(map(str, image.id))}\n")

def main():
    start_time = time.time()
    
    # Initialize processor
    processor = LargeScaleProcessor('../Data/1_binary_landscapes.txt')
    
    # Execute pipeline with memory monitoring
    print("Starting processing pipeline...")
    print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    processor.parse_input_file()
    print(f"After parsing: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    processor.find_optimal_portrait_pairs()
    print(f"After pairing: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    sorted_images = processor.sort_images()
    print(f"After sorting: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    processor.calculate_final_score(sorted_images)
    
    processor.write_output(sorted_images, '../Data/output1.txt')
    
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / len(sorted_images):.4f} seconds")

if __name__ == "__main__":
    import psutil
    process = psutil.Process()
    main()