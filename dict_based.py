from collections import defaultdict

def load_txt_to_dict(path):
    """
    Load data from a text file into a structured nested dictionary.

    Parameters:
        path (str): Path to the input text file.

    Returns:
        defaultdict: A nested dictionary where keys are row indices and 
                     values are dictionaries with row details.

    Raises:
        ValueError: If the row is malformed, `num_characteristics` is not an integer, 
                    or the number of characteristics does not match `num_characteristics`.
    """
    data = defaultdict(list)
    
    with open(path, 'r') as file:
        # Read the first line to determine the number of rows
        try:
            num_rows = int(file.readline().strip())
        except ValueError:
            raise ValueError("The first line of the file must contain an integer representing the number of rows.")
        
        # Process each subsequent line
        for index, line in enumerate(file):
            row = line.strip().split()  # Split the line into components
            
            # Check if the row has the minimum required elements
            if len(row) < 2:
                raise ValueError(f"Malformed row at index {index}: {line.strip()}")

            type_key = row[0]  # First column: type key (e.g., 'P', 'L')

            try:
                num_characteristics = int(row[1])  # Second column: number of characteristics
                characteristics = row[2:]         # Remaining columns: characteristics

                # Validate that the number of characteristics matches the specified count
                if len(characteristics) != num_characteristics:
                    raise ValueError(f"Characteristic count mismatch at row {index}. "
                                     f"Expected {num_characteristics}, got {len(characteristics)}.")
            except ValueError as e:
                raise ValueError(f"Invalid data format at row {index}: {line.strip()}. Error: {e}")

            # Add the processed row to the dictionary
            data[index].append({
                "type": type_key,
                "num_characteristics": num_characteristics,
                "characteristics": characteristics
            })

    return data

# Example usage

file_path1 = 'Data/0_example.txt'
file_path2 = 'Data/10_computable_moments.txt'
file_path3 = 'Data/11_randomizing_paintings.txt'
file_path4 = 'Data/110_oily_portraits.txt'
file_path5 = 'Data/1_binary_landscapes.txt'

output_path = 'Data/output_frames.txt'
data_dict = load_txt_to_dict(file_path1)
frames = create_frames(data_dict)
write_frames_to_file(frames, output_path)