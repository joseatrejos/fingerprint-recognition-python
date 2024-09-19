import numpy as np

def match_templates(template1, template2):
    # Example similarity measure (e.g., Euclidean distance)
    score = np.linalg.norm(template1 - template2)
    return score