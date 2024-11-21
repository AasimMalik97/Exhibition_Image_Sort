from collections import defaultdict
cimport cython

# Frame creation using a heuristic
def create_frames_fast_heuristic(object data_dict):
    cdef dict frames = {}
    cdef int frame_index = 0
    cdef set used_keys = set()
    cdef dict p_elements, l_elements
    cdef list sorted_p_keys
    cdef int key, best_pair, other_key
    cdef double max_score, score

    # Separate P and L elements
    p_elements = {key: value for key, value in data_dict.items() if value['type'] == 'P'}
    l_elements = {key: value for key, value in data_dict.items() if value['type'] == 'L'}

    # Pre-sort P elements by number of characteristics (descending)
    sorted_p_keys = sorted(p_elements.keys(), key=lambda k: len(p_elements[k]['characteristics']), reverse=True)

    while sorted_p_keys:
        key = sorted_p_keys.pop(0)
        if key in used_keys:
            continue

        # Find the closest match among the next 50 elements (reduce search space)
        best_pair = -1
        max_score = -float('inf')  # Ensure max_score is a double
        for other_key in sorted_p_keys[:50]:
            if other_key in used_keys:
                continue
            score = min_form_number_of_common_tags(
                p_elements[key]['characteristics'], p_elements[other_key]['characteristics']
            )
            if score > max_score:
                max_score = score
                best_pair = other_key

        if best_pair != -1:
            frames[frame_index] = [key, best_pair]
            used_keys.update([key, best_pair])
            sorted_p_keys.remove(best_pair)
        else:
            frames[frame_index] = [key]
            used_keys.add(key)

        frame_index += 1

    # Add L elements as single frames
    for key in l_elements:
        frames[frame_index] = [key]
        frame_index += 1

    return frames

#to modify comparision limit  , 
#for dataset with only landscapes increasing won't change anything
#for dataset with only portraits


def create_frames_fast_balanced(object data_dict, int comparison_limit=100):
    """
    Frame creation using a balanced heuristic approach with adjustable comparison limit.
    """
    cdef dict frames = {}
    cdef int frame_index = 0
    cdef set used_keys = set()
    cdef dict p_elements, l_elements
    cdef list sorted_p_keys
    cdef int key, best_pair, common_chars
    cdef double min_score

    p_elements = {key: value for key, value in data_dict.items() if value['type'] == 'P'}
    l_elements = {key: value for key, value in data_dict.items() if value['type'] == 'L'}

    # Sort based on total characteristics
    sorted_p_keys = sorted(p_elements.keys(), key=lambda k: len(p_elements[k]['characteristics']), reverse=True)

    while sorted_p_keys:
        key = sorted_p_keys.pop(0)
        if key in used_keys:
            continue

        best_pair = -1
        min_score = float('inf')

        # Adjust comparison limit
        for other_key in sorted_p_keys[:comparison_limit]:
            if other_key in used_keys:
                continue
            common_chars = len(p_elements[key]['characteristics'] & p_elements[other_key]['characteristics'])
            if common_chars < min_score:
                min_score = common_chars
                best_pair = other_key

        if best_pair != -1:
            frames[frame_index] = [key, best_pair]
            used_keys.update([key, best_pair])
            sorted_p_keys.remove(best_pair)
        else:
            frames[frame_index] = [key]
            used_keys.add(key)

        frame_index += 1

    # Add L elements as single frames
    for key in l_elements:
        frames[frame_index] = [key]
        frame_index += 1

    return frames








# Frame reordering using a greedy approach
def reorder_frames_greedy(dict frames, object data_dict):
    cdef dict frame_characteristics = {}
    cdef list frame_indices, ordered_frames
    cdef int current_frame, best_next_frame, max_similarity, candidate_frame
    cdef set characteristics
    cdef int element

    def get_frame_characteristics(frame):
        cdef set characteristics = set()
        for element in frame:
            characteristics.update(data_dict[element]["characteristics"])
        return characteristics

    # Precompute characteristics for each frame
    for frame_index, frame in frames.items():
        frame_characteristics[frame_index] = get_frame_characteristics(frame)

    frame_indices = list(frames.keys())
    ordered_frames = [frame_indices.pop(0)]

    while frame_indices:
        current_frame = ordered_frames[-1]
        best_next_frame = -1
        max_similarity = -1

        for candidate_frame in frame_indices[:50]:
            similarity = len(
                frame_characteristics[current_frame] & frame_characteristics[candidate_frame]
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_next_frame = candidate_frame

        # Ensure the best_next_frame is valid
        if best_next_frame != -1 and best_next_frame in frame_indices:
            ordered_frames.append(best_next_frame)
            frame_indices.remove(best_next_frame)
        else:
            # Fallback: Add the first remaining frame to avoid getting stuck
            ordered_frames.append(frame_indices.pop(0))

    reordered_frames = {index: frames[frame_id] for index, frame_id in enumerate(ordered_frames)}
    return reordered_frames


#lookhead to modify as well
def reorder_frames_greedy_lookahead(dict frames, object data_dict, int lookahead=16000):
    """
    Greedy frame reordering with an adjustable lookahead window for improved accuracy.
    """
    cdef dict frame_characteristics = {}
    cdef list frame_indices, ordered_frames
    cdef int current_frame, best_next_frame, max_similarity, candidate_frame
    cdef set tags1, tags2

    def get_frame_characteristics(frame):
        cdef set characteristics = set()
        for element in frame:
            characteristics.update(data_dict[element]["characteristics"])
        return characteristics

    # Precompute characteristics for each frame
    for frame_index, frame in frames.items():
        frame_characteristics[frame_index] = get_frame_characteristics(frame)

    frame_indices = list(frames.keys())
    ordered_frames = [frame_indices.pop(0)]

    while frame_indices:
        current_frame = ordered_frames[-1]
        best_next_frame = -1
        max_similarity = -1

        # Lookahead over adjustable window
        for candidate_frame in frame_indices[:lookahead]:
            tags1 = frame_characteristics[current_frame]
            tags2 = frame_characteristics[candidate_frame]
            similarity = len(tags1 & tags2)

            if similarity > max_similarity:
                max_similarity = similarity
                best_next_frame = candidate_frame

        if best_next_frame != -1:
            ordered_frames.append(best_next_frame)
            frame_indices.remove(best_next_frame)
        else:
            ordered_frames.append(frame_indices.pop(0))  # Fallback strategy

    reordered_frames = {index: frames[frame_id] for index, frame_id in enumerate(ordered_frames)}
    return reordered_frames











def calculate_score(dict frames, object data_dict):
    cdef int score = 0
    cdef list frameglasses = list(frames.values())
    cdef int i, local_score
    cdef set tags1, tags2

    for i in range(len(frameglasses) - 1):
        tags1 = combine_tags(frameglasses[i], data_dict)
        tags2 = combine_tags(frameglasses[i + 1], data_dict)
        local_score = min_form_number_of_common_tags(tags1, tags2)
        score += local_score

    return score

cdef set combine_tags(list frame, object data_dict):
    cdef set combined_tags = set()
    cdef int element
    for element in frame:
        combined_tags.update(data_dict[element]["characteristics"])
    return combined_tags

cdef int min_form_number_of_common_tags(set tags1, set tags2):
    cdef int tags_in_tags1_not_in_tags2 = len(tags1 - tags2)
    cdef int tags_in_tags2_not_in_tags1 = len(tags2 - tags1)
    cdef int common_tags = len(tags1 & tags2)
    return min(tags_in_tags1_not_in_tags2, tags_in_tags2_not_in_tags1, common_tags)
