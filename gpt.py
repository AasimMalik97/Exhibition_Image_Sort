import json


def process_large_file(file_path, output_path):
    """Processes a file to maximize Global Robotic Satisfaction."""
    total_score = 0
    portrait_images = []
    landscape_images = []
    final_frames = []

    def parse_line(line):
        """Parse each line into a dictionary."""
        parts = line.strip().split()
        orientation = parts[0]
        tags = parts[2:]  # Keep tags as a list for JSON compatibility
        return {"orientation": orientation, "tags": tags}

    def calculate_local_satisfaction(tags1, tags2):
        """Calculate Local Robotic Satisfaction between two frames."""
        set_tags1 = set(tags1)
        set_tags2 = set(tags2)
        common_tags = len(set_tags1 & set_tags2)
        tags_in_1_not_in_2 = len(set_tags1 - set_tags2)
        tags_in_2_not_in_1 = len(set_tags2 - set_tags1)
        return min(common_tags, tags_in_1_not_in_2, tags_in_2_not_in_1)

    def calculate_global_satisfaction(frames):
        """Calculate the Global Robotic Satisfaction for a sequence of frames."""
        global_score = 0
        for i in range(len(frames) - 1):
            global_score += calculate_local_satisfaction(frames[i]["tags"], frames[i + 1]["tags"])
        return global_score

    # Parse input file
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for idx, line in enumerate(lines[1:]):  # Skip the first line (batch count)
            image = parse_line(line)
            image["id"] = [idx]  # Ensure ID is a list
            if image["orientation"] == "P":
                portrait_images.append(image)
            elif image["orientation"] == "L":
                landscape_images.append(image)

    # Combine portrait images into frames of two
    while len(portrait_images) > 1:
        img1 = portrait_images.pop(0)  # Remove the first portrait
        img2 = portrait_images.pop(0)  # Remove the next portrait
        combined_tags = list(set(img1["tags"]) | set(img2["tags"]))  # Union of tags
        final_frames.append({
            "id": img1["id"] + img2["id"],  # Combine IDs
            "orientation": "P",
            "tags": combined_tags
        })

    # Add standalone landscape images as frames
    while len(landscape_images) > 0:
        img = landscape_images.pop(0)  # Remove the first landscape
        final_frames.append({
            "id": img["id"],
            "orientation": "L",
            "tags": img["tags"]
        })

    # Reorder frames to maximize Global Robotic Satisfaction
    def reorder_frames(frames):
        sorted_frames = [frames.pop(0)]  # Start with the first frame
        while frames:
            best_next_frame = max(frames, key=lambda f: calculate_local_satisfaction(sorted_frames[-1]["tags"], f["tags"]))
            sorted_frames.append(best_next_frame)
            frames.remove(best_next_frame)
        return sorted_frames

    final_frames = reorder_frames(final_frames)

    # Calculate the total Global Robotic Satisfaction
    total_score = calculate_global_satisfaction(final_frames)

    # Write output to file
    with open(output_path, 'w') as out_file:
        frame_count = len(final_frames)  # Count the number of frames
        out_file.write(f"Global Robotic Satisfaction: {total_score}\n")  # Write the total score
        out_file.write(f"Number of Frames: {frame_count}\n")  # Write the number of frames
        for frame in final_frames:
            frame_id = " ".join(map(str, frame["id"]))  # Join IDs as space-separated strings
            out_file.write(f"{frame_id}\n")

    print(f"Processing completed. Global Robotic Satisfaction: {total_score}, Number of Frames: {frame_count}")

    # Save detailed JSON output
    json_output_path = output_path.replace('.txt', '.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(final_frames, json_file, indent=4)
    print(f"Detailed JSON written to {json_output_path}")


# File paths
input_file = "./data/10_computable_moments.txt"
output_file = "output_frames.txt"

# Process the file
process_large_file(input_file, output_file)
