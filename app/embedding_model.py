"""Word embeddings with spaCy, adapted from Module 2 Practical 3 (Word Embeddings)."""

import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_lg"):
        """Load the spaCy model once. This is slow (a few seconds), so it
        should happen when the app starts, not on every request."""
        self.nlp = spacy.load(model_name)

    def calculate_embedding(self, word):
        """Return the embedding vector for a word as a plain Python list,
        or None if the word is not in the model's vocabulary."""
        doc = self.nlp(word)
        if not doc.has_vector:  # unknown word: spaCy would return all zeros
            return None
        # doc.vector is a NumPy float32 array, which JSON can't serialize,
        # so convert it to a list of regular floats.
        return doc.vector.tolist()

    def calculate_similarity(self, word1, word2):
        """Cosine similarity between two words' embeddings (Practical 3)."""
        return float(self.nlp(word1).similarity(self.nlp(word2)))