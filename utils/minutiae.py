from scipy.spatial import distance

def match_minutiaes(minutiae_A, minutiae_B, distance_threshold=10, angle_threshold=15):
    """
    Match minutiae points between two sets based on Euclidean distance and orientation.
    """
    matched_minutiae = 0
    used_B_indices = set()

    for minutia_A in minutiae_A:
        (x_A, y_A, angle_A) = minutia_A
        for i, minutia_B in enumerate(minutiae_B):
            if i in used_B_indices:
                continue  # Skip already matched minutiae
            (x_B, y_B, angle_B) = minutia_B

            if distance.euclidean((x_A, y_A), (x_B, y_B)) < distance_threshold and abs(angle_A - angle_B) < angle_threshold:
                matched_minutiae += 1
                used_B_indices.add(i)
                break

    return matched_minutiae

def compute_dissimilarity(minutiae_A, minutiae_B, distance_threshold=10, angle_threshold=15):
    """
    Compute the dissimilarity index between two fingerprints based on minutiae.
    """
    N_m = match_minutiaes(minutiae_A, minutiae_B, distance_threshold, angle_threshold)
    N_t = len(minutiae_A) + len(minutiae_B)

    if N_t == 0:
        return 1.0  # Avoid division by zero if no minutiae

    dissimilarity_index = 1 - (2 * N_m / N_t)
    return dissimilarity_index
