from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

def generate_normalized_bow(texts):
    """
    Generates a normalized bag-of-words representation from an array of texts.

    Parameters:
        texts (list of str): List of input texts.

    Returns:
        np.ndarray: Normalized bag-of-words matrix.
        list: List of feature names (words).
    """
    # Create a CountVectorizer instance to compute the BoW
    vectorizer = CountVectorizer()
    # Fit and transform the input texts
    bow_matrix = vectorizer.fit_transform(texts)
    # Normalize the matrix
    normalized_bow = normalize(bow_matrix, norm='l2', axis=1)
    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()
    print(normalized_bow)
    return normalized_bow.toarray()