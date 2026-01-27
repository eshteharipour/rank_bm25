#!/usr/bin/env python

"""
Rank-BM25: A Collection of BM25 and Information Retrieval Algorithms

Original BM25 implementations by Dorian Brown (https://github.com/dorianbrown/rank_bm25)
Additional algorithms (TF-IDF, Jaccard, IDF) by S. Eshteharipour

Licensed under the Apache License, Version 2.0
Based on: Trotman et al, Improvements to BM25 and Language Models Examined
"""

import math
from multiprocessing import Pool, cpu_count

import numpy as np

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        """Deprecated."""

        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

    def search(
        self, queries: np.ndarray | list, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(queries, np.ndarray):
            assert queries.ndim == 2
        else:
            assert isinstance(queries, list), type(queries)
            assert isinstance(queries[0], list), type(queries[0])
            assert isinstance(queries[0][0], str), type(queries[0][0])

        all_scores = []
        all_indices = []
        for q in queries:
            scores = self.get_scores(q)
            top_n = np.argsort(scores)[::-1][:k]
            scores_srt = np.array([scores[i] for i in top_n])

            all_scores.append(scores_srt)
            all_indices.append(top_n)

        all_scores = np.array(all_scores)
        all_indices = np.array(all_indices)

        return all_scores, all_indices


class IDF(BM25):
    def __init__(self, corpus, tokenizer=None):
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculate IDF using the standard IDF formula: log(N/df)
        where N is the total number of documents and df is document frequency
        """
        self.idf = {}
        for word, freq in nd.items():
            # Standard IDF calculation
            idf = math.log(
                self.corpus_size / (freq + 1)
            )  # +1 to avoid division by zero
            self.idf[word] = idf

    def get_scores(self, query):
        """
        Calculate IDF scores: IDF
        - IDF is pre-calculated in _calc_idf
        - No length normalization or parameters like in BM25
        """
        score = np.zeros(self.corpus_size)
        for q in query:
            # Get term frequency for query term across all documents
            # IDF score = term frequency × inverse document frequency
            score += self.idf.get(q) or 0
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate IDF scores for a subset of documents
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        for q in query:
            # IDF score = term frequency × inverse document frequency
            score += self.idf.get(q) or 0
        return score.tolist()


class TFIDF_A(BM25):  # Grok
    def __init__(self, corpus, tokenizer=None):
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculate IDF using the standard TF-IDF formula: log(N/df)
        where N is the total number of documents and df is document frequency
        """
        self.idf = {}
        for word, freq in nd.items():
            # Standard TF-IDF IDF calculation
            idf = math.log(
                self.corpus_size / (freq + 1)
            )  # +1 to avoid division by zero
            self.idf[word] = idf

    def get_scores(self, query):
        """
        Calculate TF-IDF scores: TF × IDF
        - TF is the raw term frequency
        - IDF is pre-calculated in _calc_idf
        - No length normalization or parameters like in BM25
        """
        score = np.zeros(self.corpus_size)
        for q in query:
            # Get term frequency for query term across all documents
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            # TF-IDF score = term frequency × inverse document frequency
            score += q_freq * (self.idf.get(q) or 0)
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate TF-IDF scores for a subset of documents
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            # TF-IDF score = term frequency × inverse document frequency
            score += q_freq * (self.idf.get(q) or 0)
        return score.tolist()


class TFIDF_B(BM25):  # Claude
    def __init__(self, corpus, tokenizer=None, smooth_idf=True, norm=None):
        """
        Initialize the TF-IDF scoring model.

        Args:
            corpus: List of documents, where each document is a list of tokens
            tokenizer: Optional tokenizer function to process raw text
            smooth_idf: Whether to add 1 to document frequencies to prevent division by zero
            norm: Normalization method ('l1', 'l2', or None)
        """
        self.smooth_idf = smooth_idf
        self.norm = norm
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculate Inverse Document Frequency for each term.

        Args:
            nd: Dictionary mapping terms to their document frequencies
        """
        for word, freq in nd.items():
            # Standard TF-IDF formula with optional smoothing
            if self.smooth_idf:
                idf = math.log((self.corpus_size + 1) / (freq + 1)) + 1
            else:
                idf = math.log(self.corpus_size / freq)

            self.idf[word] = idf

    def get_scores(self, query):
        """
        Calculate TF-IDF scores for a query against all documents in the corpus.

        Args:
            query: List of query terms

        Returns:
            numpy array of scores for each document
        """
        score = np.zeros(self.corpus_size)

        # Calculate TF-IDF for each term in the query
        for q in query:
            if q not in self.idf:
                continue

            # Term frequency in each document (using raw count as TF)
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])

            # TF-IDF = term frequency * inverse document frequency
            score += q_freq * self.idf.get(q, 0)

        # Apply normalization if specified
        if self.norm == "l1":
            # L1 normalization (Manhattan distance)
            norm_factors = np.sum(np.abs(score)) or 1  # Avoid division by zero
            score = score / norm_factors
        elif self.norm == "l2":
            # L2 normalization (Euclidean distance)
            norm_factors = np.sqrt(np.sum(score**2)) or 1  # Avoid division by zero
            score = score / norm_factors

        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate TF-IDF scores between query and subset of documents.

        Args:
            query: List of query terms
            doc_ids: List of document indices to score against

        Returns:
            List of scores for specified documents
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))

        # Calculate TF-IDF for each term in the query
        for q in query:
            if q not in self.idf:
                continue

            # Term frequency in selected documents
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])

            # TF-IDF = term frequency * inverse document frequency
            score += q_freq * self.idf.get(q, 0)

        # Apply normalization if specified
        if self.norm == "l1":
            # L1 normalization (Manhattan distance)
            norm_factors = np.sum(np.abs(score)) or 1  # Avoid division by zero
            score = score / norm_factors
        elif self.norm == "l2":
            # L2 normalization (Euclidean distance)
            norm_factors = np.sqrt(np.sum(score**2)) or 1  # Avoid division by zero
            score = score / norm_factors

        return score.tolist()


class TFIDF_C(BM25):
    def __init__(self, corpus, tokenizer=None):
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        self.idf = {}
        for word, freq in nd.items():
            idf = math.log(
                self.corpus_size / (freq + 1)
            )  # +1 to avoid division by zero
            self.idf[word] = idf

    def get_scores(self, query):
        # Compute unnormalized TF-IDF score
        score = np.zeros(self.corpus_size)

        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += q_freq * (self.idf.get(q) or 0)

        # === NEW: L2 Normalize each document's TF-IDF vector ===
        norms = np.zeros(self.corpus_size)
        for i, doc in enumerate(self.doc_freqs):
            tfidf_squared_sum = sum(
                (tf * self.idf.get(term, 0)) ** 2 for term, tf in doc.items()
            )
            norms[i] = (
                math.sqrt(tfidf_squared_sum) if tfidf_squared_sum > 0 else 1.0
            )  # avoid div-by-zero

        normalized_score = score / norms
        return normalized_score


class Jaccard(BM25):
    def __init__(self, corpus, tokenizer=None):
        """
        Initialize the Jaccard similarity scoring model.

        Args:
            corpus: List of documents, where each document is a list of tokens
            tokenizer: Optional tokenizer function to process raw text
        """
        super().__init__(corpus, tokenizer)
        # Jaccard doesn't use IDF, so we'll override it with an empty dict
        self.idf = {}

    def _calc_idf(self, nd):
        """
        Jaccard similarity doesn't use IDF, so this is a no-op.
        Required to satisfy the abstract base class.
        """
        pass

    def get_scores(self, query):
        """
        Calculate Jaccard similarity scores for a query against all documents.
        Jaccard similarity = |A ∩ B| / |A ∪ B|

        Args:
            query: List of query terms

        Returns:
            numpy array of Jaccard similarity scores for each document
        """
        scores = np.zeros(self.corpus_size)
        query_set = set(query)

        for i, doc_freq in enumerate(self.doc_freqs):
            doc_set = set(doc_freq.keys())

            # Calculate intersection and union
            intersection = len(query_set & doc_set)
            union = len(query_set | doc_set)

            # Avoid division by zero
            if union == 0:
                scores[i] = 0.0
            else:
                scores[i] = intersection / union

        return scores

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate Jaccard similarity scores for a query against a subset of documents.

        Args:
            query: List of query terms
            doc_ids: List of document indices to score against

        Returns:
            List of Jaccard similarity scores for specified documents
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        scores = np.zeros(len(doc_ids))
        query_set = set(query)

        for i, doc_id in enumerate(doc_ids):
            doc_set = set(self.doc_freqs[doc_id].keys())

            # Calculate intersection and union
            intersection = len(query_set & doc_set)
            union = len(query_set | doc_set)

            # Avoid division by zero
            if union == 0:
                scores[i] = 0.0
            else:
                scores[i] = intersection / union

        return scores.tolist()


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score.tolist()


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
                # * q_freq
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
                # * q_freq
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            # idf = math.log((self.corpus_size + 1) / freq)
            idf = math.log(self.corpus_size + 1) - math.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score.tolist()


# TODO:
# BM25Adpt and BM25T are a bit more complicated than the previous algorithms here. Here a term-specific k1
# parameter is calculated before scoring is done


class BM25Adpt(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score


class BM25T(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score
