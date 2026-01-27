import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import numpy as np

from rank_bm25 import (
    BM25L,
    IDF,
    TFIDF_A,
    TFIDF_B,
    TFIDF_C,
    BM25Okapi,
    BM25Plus,
    Jaccard,
)

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

# Original algorithms
bm25_algs = [
    BM25Okapi(tokenized_corpus),
    BM25L(tokenized_corpus),
    BM25Plus(tokenized_corpus),
]

# New algorithms
new_algs = [
    IDF(tokenized_corpus),
    TFIDF_A(tokenized_corpus),
    TFIDF_B(tokenized_corpus),
    TFIDF_C(tokenized_corpus),
    Jaccard(tokenized_corpus),
]

all_algs = bm25_algs + new_algs


def test_corpus_loading():
    """Test that corpus is loaded correctly for all algorithms"""
    for alg in all_algs:
        assert alg.corpus_size == 3
        assert alg.avgdl == 5
        assert alg.doc_len == [4, 6, 5]


def tokenizer(doc):
    return doc.split(" ")


def test_tokenizer():
    """Test that custom tokenizer works"""
    bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
    assert bm25.corpus_size == 3
    assert bm25.avgdl == 5
    assert bm25.doc_len == [4, 6, 5]


def test_get_scores():
    """Test that get_scores returns correct shape and type"""
    query = "windy London"
    tokenized_query = query.split(" ")

    for alg in all_algs:
        scores = alg.get_scores(tokenized_query)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert scores.dtype in [np.float64, np.float32, float]


def test_get_batch_scores():
    """Test batch scoring for subset of documents"""
    query = "windy London"
    tokenized_query = query.split(" ")
    doc_ids = [0, 2]

    for alg in all_algs:
        scores = alg.get_batch_scores(tokenized_query, doc_ids)
        assert isinstance(scores, list)
        assert len(scores) == 2


def test_get_top_n():
    """Test that get_top_n returns correct documents"""
    query = "windy London"
    tokenized_query = query.split(" ")

    for alg in bm25_algs:  # Only test original BM25 variants
        top_doc = alg.get_top_n(tokenized_query, corpus, n=1)
        assert isinstance(top_doc, list)
        assert len(top_doc) == 1
        assert top_doc[0] in corpus


def test_search_method():
    """Test the new search method for batch queries"""
    queries = [["windy", "London"], ["weather", "today"], ["Hello", "man"]]

    for alg in all_algs:
        scores, indices = alg.search(queries, k=2)

        assert isinstance(scores, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert scores.shape == (3, 2)  # 3 queries, top 2 results each
        assert indices.shape == (3, 2)

        # Check that indices are valid
        assert np.all(indices >= 0)
        assert np.all(indices < alg.corpus_size)


def test_tfidf_normalization():
    """Test TF-IDF with different normalization options"""
    query = ["windy", "London"]

    # Test without normalization
    tfidf_none = TFIDF_B(tokenized_corpus, norm=None)
    scores_none = tfidf_none.get_scores(query)

    # Test with L1 normalization
    tfidf_l1 = TFIDF_B(tokenized_corpus, norm="l1")
    scores_l1 = tfidf_l1.get_scores(query)

    # Test with L2 normalization
    tfidf_l2 = TFIDF_B(tokenized_corpus, norm="l2")
    scores_l2 = tfidf_l2.get_scores(query)

    # Normalized scores should be different from unnormalized
    assert not np.allclose(scores_none, scores_l1)
    assert not np.allclose(scores_none, scores_l2)

    # L1 normalized sum should be close to 1
    assert np.abs(np.sum(np.abs(scores_l1)) - 1.0) < 1e-10

    # L2 normalized should have unit norm
    assert np.abs(np.sqrt(np.sum(scores_l2**2)) - 1.0) < 1e-10


def test_jaccard_similarity():
    """Test Jaccard similarity scores are in [0, 1] range"""
    query = ["windy", "London"]
    jaccard = Jaccard(tokenized_corpus)
    scores = jaccard.get_scores(query)

    assert np.all(scores >= 0)
    assert np.all(scores <= 1)

    # Test exact match
    exact_query = "It is quite windy in London".split()
    scores_exact = jaccard.get_scores(exact_query)
    assert scores_exact[1] == 1.0  # Should be perfect match with doc 1


def test_idf_scores():
    """Test IDF scoring (query-independent document frequencies)"""
    query = ["windy", "London"]
    idf = IDF(tokenized_corpus)
    scores = idf.get_scores(query)

    # IDF scores should be based on query terms only
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3


def test_tfidf_variants_consistency():
    """Test that all TF-IDF variants produce valid scores"""
    query = ["windy", "London"]

    tfidf_a = TFIDF_A(tokenized_corpus)
    tfidf_b = TFIDF_B(tokenized_corpus)
    tfidf_c = TFIDF_C(tokenized_corpus)

    scores_a = tfidf_a.get_scores(query)
    scores_b = tfidf_b.get_scores(query)
    scores_c = tfidf_c.get_scores(query)

    # All should return valid numpy arrays
    assert isinstance(scores_a, np.ndarray)
    assert isinstance(scores_b, np.ndarray)
    assert isinstance(scores_c, np.ndarray)

    # All should have same length
    assert len(scores_a) == len(scores_b) == len(scores_c) == 3


def test_empty_query():
    """Test behavior with empty query"""
    empty_query = []

    for alg in all_algs:
        scores = alg.get_scores(empty_query)
        # Empty query should return zero scores
        assert np.all(scores == 0)


def test_query_not_in_corpus():
    """Test behavior when query terms are not in corpus"""
    query = ["xyz", "abc", "notincorpus"]

    for alg in all_algs:
        scores = alg.get_scores(query)
        # Should return valid scores (likely zeros or very low)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3


def test_search_with_single_query():
    """Test search method with single query"""
    queries = [["windy", "London"]]

    for alg in all_algs:
        scores, indices = alg.search(queries, k=2)
        assert scores.shape == (1, 2)
        assert indices.shape == (1, 2)


def test_search_k_larger_than_corpus():
    """Test search when k is larger than corpus size"""
    queries = [["windy", "London"]]
    k = 10  # Larger than corpus size (3)

    for alg in all_algs:
        scores, indices = alg.search(queries, k=k)
        # Should return all documents (corpus_size)
        assert scores.shape == (1, 3)
        assert indices.shape == (1, 3)
