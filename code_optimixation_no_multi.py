import json
import time
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

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
        self.chunk_size = 1000

    def _process_line(self, idx: int, line: str) -> ImageData:
        if not line.strip():
            return None
        
        parts = line.split()
        return ImageData(
            id=[idx],
            type=parts[0],
            tags=set(parts[2:]),
            tag_length=int(parts[1])
        )

    def parse_input_file(self) -> None:
        with open(self.file_path, 'r') as file:
            next(file)  # Skip header
            for idx, line in enumerate(file):
                image_data = self._process_line(idx, line)
                if image_data:
                    if image_data.type == "P":
                        self.portrait_images.append(image_data)
                    else:
                        self.landscape_images.append(image_data)

    def find_optimal_portrait_pairs(self) -> None:
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

        self.frame_portrait_images = Frame_Portrait_Images

    def build_tag_index(self, final_list: List[ImageData]) -> None:
        self.tag_to_images.clear()
        self.image_to_tags.clear()

        for idx, image in enumerate(final_list):
            for tag in image.tags:
                self.tag_to_images[tag].add(idx)
            self.image_to_tags[idx] = image.tags

    def sort_images_fast(self) -> List[ImageData]:
        final_list = self.frame_portrait_images + self.landscape_images
        if not final_list:
            return []

        self.build_tag_index(final_list)

        sorted_images = [final_list[0]]
        used_indices = {0}
        similarity_cache = {}

        while len(sorted_images) < len(final_list):
            print(f"Images sorted: {len(sorted_images)} / {len(final_list)}", end='\r')
            
            current_tags = sorted_images[-1].tags
            candidate_indices = set()

            for tag in current_tags:
                candidate_indices.update(self.tag_to_images[tag])
            candidate_indices -= used_indices

            if not candidate_indices:
                for i in range(len(final_list)):
                    if i not in used_indices:
                        sorted_images.append(final_list[i])
                        used_indices.add(i)
                        break
                continue

            best_similarity = float('-inf')
            best_idx = None

            for idx in candidate_indices:
                cache_key = (frozenset(current_tags), frozenset(final_list[idx].tags))
                if cache_key not in similarity_cache:
                    similarity_cache[cache_key] = len(current_tags & final_list[idx].tags)

                similarity = similarity_cache[cache_key]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = idx

            if best_idx is not None:
                sorted_images.append(final_list[best_idx])
                used_indices.add(best_idx)

        return sorted_images

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

        return {
            "total_score": total_score,
            "average_score": round(average_score, 2),
            "total_transitions": num_transitions,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity
        }

    def write_output(self, sorted_images: List[ImageData], output_path: str) -> None:
        scores = self.calculate_final_score(sorted_images)

        with open(output_path, 'w') as file:
            file.write(f"{len(sorted_images)}\n")
            for image in sorted_images:
                file.write(f"{' '.join(map(str, image.id))}\n")


def main():
    start_time = time.time()

    processor = LargeScaleProcessor('../Data/1_binary_landscapes.txt')  # Replace with your input file path
    print("Parsing input file...")
    processor.parse_input_file()

    print("Finding optimal portrait pairs...")
    processor.find_optimal_portrait_pairs()

    print("Sorting images...")
    sorted_images = processor.sort_images_fast()

    scores = processor.calculate_final_score(sorted_images)
    print("\nSorting Statistics:")
    print(f"Total Score: {scores['total_score']}")
    print(f"Average Score per Transition: {scores['average_score']}")
    print(f"Total Transitions: {scores['total_transitions']}")
    print(f"Minimum Similarity: {scores['min_similarity']}")
    print(f"Maximum Similarity: {scores['max_similarity']}")

    print("\nWriting output...")
    processor.write_output(sorted_images, "../Data/output1.txt")  # Replace with your output file path

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
