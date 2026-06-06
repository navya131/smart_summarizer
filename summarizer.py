import nltk
import spacy
import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fuzzywuzzy import fuzz

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Stopwords
stop_words = set(stopwords.words("english"))

# -----------------------------------
# CLEAN TEXT
# -----------------------------------
def clean_text(text):

    doc = nlp(text.lower())

    cleaned_words = []

    for token in doc:

        if token.text not in stop_words and token.is_alpha:

            cleaned_words.append(token.lemma_)

    return " ".join(cleaned_words)

# -----------------------------------
# SUMMARIZATION
# -----------------------------------
def summarize_text(text, summary_ratio=0.4):

    sentences = [s.strip() for s in text.split('.') if s.strip()]

    if len(sentences) <= 2:
        return text

    # Clean all sentences
    cleaned_sentences = [
        clean_text(sentence)
        for sentence in sentences
    ]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()

    sentence_vectors = vectorizer.fit_transform(
        cleaned_sentences
    )

    # Sentence Scores
    sentence_scores = np.array(
        sentence_vectors.sum(axis=1)
    ).flatten()

    # Summary sentence count
    summary_length = max(
        1,
        int(len(sentences) * summary_ratio)
    )

    # Top ranked sentence indexes
    top_sentence_indexes = sentence_scores.argsort()[
        -summary_length:
    ]

    # Keep original order
    top_sentence_indexes = sorted(
        top_sentence_indexes
    )

    # Build summary
    summary = " ".join([
        sentences[i]
        for i in top_sentence_indexes
    ])

    return summary

#
# SIMILARITY SCORE
# 
def calculate_similarity(original, summary):

    # Fuzzy Similarity
    fuzzy_score = fuzz.token_set_ratio(
        original,
        summary
    )

    # Cosine Similarity
    documents = [original, summary]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    cosine_score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    cosine_score = round(
        cosine_score * 100,
        2
    )

    # Final Average Score
    final_score = round(
        (fuzzy_score + cosine_score) / 2,
        2
    )

    return {
        "fuzzy_score": fuzzy_score,
        "cosine_score": cosine_score,
        "final_score": final_score
    }
