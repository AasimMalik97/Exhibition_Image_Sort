import sys
import numpy as np
from random import shuffle, randint
import os
from typing import List, Dict

def read_input(file_path):
    
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None
    
    with open(file_path, 'r') as file:
        paintings = []
        N = int(file.readline().strip())
        print(f"Reading {N} paintings from the file.")
        for i in range(N):
            data = file.readline().strip().split()
            paintings.append({
                'id': i,
                'type': data[0],
                'tags': set(data[2:])  # Using set to handle unique tags
            })
        print("Finished reading input data.")
    return paintings

def arrange_paintings(paintings):
    landscapes = [p for p in paintings if p['type'] == 'L']
    portraits = [p for p in paintings if p['type'] == 'P']
    print(f"Arranging {len(landscapes)} landscapes and {len(portraits)} portraits into frameglasses.")

    portrait_pairs = []
    used_portraits = set()
    for i in range(len(portraits)):
        for j in range(i + 1, len(portraits)):
            if i not in used_portraits and j not in used_portraits:
                if len(portraits[i]['tags'] & portraits[j]['tags']) > 0:
                    portrait_pairs.append((portraits[i]['id'], portraits[j]['id']))
                    used_portraits.update([i, j])
    print(f"Paired {len(portrait_pairs)} sets of portraits.")

    remaining_portraits = [p['id'] for i, p in enumerate(portraits) if i not in used_portraits]
    while len(remaining_portraits) > 1:
        portrait_pairs.append((remaining_portraits.pop(), remaining_portraits.pop()))
    if remaining_portraits:
        portrait_pairs.append((remaining_portraits.pop(),))  # Single portrait in a frame

    print("Completed arrangement of paintings.")
    return landscapes, portrait_pairs

def genetic_algorithm_solver(frameglasses, iterations=1000, population_size=50, mutation_rate=0.05):
    print("Starting genetic algorithm to optimize frameglass sequence.")
    population = [list(frameglasses) for _ in range(population_size)]
    for p in population:
        shuffle(p)
    
    best_solution = population[0]
    best_grs = calculate_grs(best_solution)

    for iteration in range(iterations):
        print(f"Optimization iteration {iteration + 1}/{iterations}")
        new_population = []
        for i in range(population_size):
            # Placeholder for genetic algorithm operations: mutation and crossover
            new_solution = list(best_solution)
            shuffle(new_solution)
            new_grs = calculate_grs(new_solution)
            if new_grs > best_grs:
                best_solution, best_grs = new_solution, new_grs
            new_population.append(new_solution)
        population = new_population

    print("Genetic algorithm optimization completed.")
    return best_solution

def calculate_grs(sequence):
    # Placeholder function to calculate GRS
    return sum(randint(1, 10) for _ in range(len(sequence) - 1))

def write_output(frameglasses, file_path='output.txt'):
    print(f"Writing output to {file_path}.")
    with open(file_path, 'w') as file:
        file.write(f"{len(frameglasses)}\n")
        for fg in frameglasses:
            file.write(" ".join(map(str, fg)) + "\n")
    print("Output writing complete.")
    
def calculate_final_score(self, sorted_images):
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


def main(input_file):
    paintings = read_input(input_file)
    landscapes, portrait_pairs = arrange_paintings(paintings)
    frameglasses = landscapes + portrait_pairs
    optimized_sequence = genetic_algorithm_solver(frameglasses)
    calculate_final_score(optimized_sequence)
    write_output(optimized_sequence)

if __name__ == '__main__':
    main('../../Data/11_randomizing_paintings.txt')
