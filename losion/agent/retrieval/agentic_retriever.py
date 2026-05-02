"""
Agentic Retriever — Multi-round retrieval with confidence-based query refinement.

Inspired by:
- Agentic RAG (Multiple papers, 2024-2025): Traditional RAG is passive
  (retrieve once, generate). Agentic RAG is active: the agent decides
  WHEN to retrieve, WHAT queries to use, and HOW to combine multiple rounds.
- CRP-RAG (2024): Uses reasoning graphs for complex query reasoning.
- OPEN-RAG (2024): Enhanced retrieval with self-reflection on quality.
- SR-RAG: Selective retrieval — don't always retrieve, only when needed.

Current Losion WebSearchInterface does single-round search. This module
replaces it with multi-round retrieval that:

1. Initial search from user query
2. Evaluate retrieval quality (using confidence signals)
3. If insufficient, reformulate query based on partial results
4. Re-search with refined query
5. Synthesize all results

Key design:
    Query → Round 1 (initial) → Quality Check → Round 2 (refined) → ... → Synthesis

Integration with Losion:
- Uses Tri-Jalur routing weights for retrieval quality assessment
- Uses CalibrationEngine tool trust for search backend selection
- Integrates with SignalExtractor for knowledge sufficiency checks
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from losion.agent.tools.web_search import WebSearchInterface, SearchResult, SearchConfig

logger = logging.getLogger(__name__)


class RetrievalQuality(Enum):
    """Quality assessment of a retrieval round."""

    SUFFICIENT = "sufficient"          # Results fully answer the query
    PARTIAL = "partial"                # Some relevant results, need more
    INSUFFICIENT = "insufficient"      # Results are not helpful
    IRRELEVANT = "irrelevant"          # Results don't match the query at all


@dataclass
class RetrievalRound:
    """A single round of retrieval in the multi-round process.

    Attributes:
        round_number: Which round this is (1-indexed).
        query: The search query used in this round.
        results: Search results from this round.
        quality: Assessed quality of the results.
        quality_score: Numerical quality score [0.0, 1.0].
        refinement_reason: Why the query was refined (for rounds > 1).
        timestamp: When this round was executed.
    """

    round_number: int = 1
    query: str = ""
    results: List[SearchResult] = field(default_factory=list)
    quality: RetrievalQuality = RetrievalQuality.INSUFFICIENT
    quality_score: float = 0.0
    refinement_reason: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AgenticRetriever:
    """Multi-round retrieval with confidence-based query refinement.

    Unlike single-round WebSearchInterface.search(), this retriever:
    1. Searches with the initial query
    2. Evaluates result quality
    3. Refines the query if results are insufficient
    4. Re-searches with the refined query
    5. Synthesizes all results across rounds

    Query refinement strategies:
    - ADD_CONTEXT: Add domain-specific terms from initial results
    - REPHRASE: Rephrase the query using synonyms from results
    - DECOMPOSE: Split complex queries into sub-queries
    - NARROW: Add restrictive terms to narrow broad results
    - BROADEN: Remove specific terms to broaden narrow results

    Usage:
        retriever = AgenticRetriever(web_search=my_search)
        result = retriever.retrieve(
            query="What is quantum computing?",
            max_rounds=3,
        )
        # result = synthesized results from multiple rounds

    Args:
        web_search: WebSearchInterface for actual search execution.
        max_rounds: Maximum retrieval rounds per query.
        quality_threshold: Stop when quality exceeds this.
        enable_decomposition: Whether to decompose complex queries.
        enable_synthesis: Whether to synthesize results across rounds.
    """

    # Query refinement strategies
    REFINE_ADD_CONTEXT = "add_context"
    REFINE_REPHRASE = "rephrase"
    REFINE_DECOMPOSE = "decompose"
    REFINE_NARROW = "narrow"
    REFINE_BROADEN = "broaden"

    def __init__(
        self,
        web_search: Optional[WebSearchInterface] = None,
        max_rounds: int = 3,
        quality_threshold: float = 0.6,
        enable_decomposition: bool = True,
        enable_synthesis: bool = True,
    ) -> None:
        self.web_search = web_search or WebSearchInterface()
        self.max_rounds = max_rounds
        self.quality_threshold = quality_threshold
        self.enable_decomposition = enable_decomposition
        self.enable_synthesis = enable_synthesis

    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        max_rounds: Optional[int] = None,
        initial_confidence: Optional[float] = None,
    ) -> Tuple[List[SearchResult], List[RetrievalRound]]:
        """Execute multi-round retrieval with query refinement.

        Args:
            query: The search query.
            domain: Domain classification for context.
            max_rounds: Override maximum rounds.
            initial_confidence: Model confidence (used for quality assessment).

        Returns:
            Tuple of (synthesized_results, retrieval_rounds).
        """
        rounds: List[RetrievalRound] = []
        all_results: List[SearchResult] = []
        effective_max = max_rounds or self.max_rounds
        current_query = query

        for round_num in range(1, effective_max + 1):
            # === Execute search ===
            results = self.web_search.search(query=current_query)

            # === Assess quality ===
            quality, score = self._assess_quality(
                query=query, results=results, domain=domain,
                initial_confidence=initial_confidence,
            )

            # === Record round ===
            retrieval_round = RetrievalRound(
                round_number=round_num,
                query=current_query,
                results=results,
                quality=quality,
                quality_score=score,
            )
            rounds.append(retrieval_round)
            all_results.extend(results)

            # === Check if sufficient ===
            if quality in (RetrievalQuality.SUFFICIENT,) or score >= self.quality_threshold:
                logger.info(
                    f"Retrieval round {round_num}: quality={quality.value}, "
                    f"score={score:.2f} — sufficient, stopping."
                )
                break

            # === Refine query for next round ===
            if round_num < effective_max:
                refined_query, strategy = self._refine_query(
                    original_query=query,
                    current_query=current_query,
                    results=results,
                    quality=quality,
                    domain=domain,
                )
                retrieval_round.refinement_reason = (
                    f"Strategy: {strategy}. "
                    f"Quality was {quality.value} ({score:.2f}), refining query."
                )
                current_query = refined_query

                logger.info(
                    f"Retrieval round {round_num}: quality={quality.value}, "
                    f"score={score:.2f} — refining with strategy {strategy}."
                )

        # === Synthesize results ===
        if self.enable_synthesis:
            synthesized = self._synthesize_results(all_results, query)
        else:
            synthesized = all_results

        return synthesized, rounds

    def _assess_quality(
        self,
        query: str,
        results: List[SearchResult],
        domain: Optional[str],
        initial_confidence: Optional[float] = None,
    ) -> Tuple[RetrievalQuality, float]:
        """Assess the quality of search results for the given query.

        Uses multiple heuristics:
        1. Result count (no results = insufficient)
        2. Relevance scores (average relevance)
        3. Query coverage (do results cover key query terms?)
        4. Content richness (do results have substantial snippets?)

        Args:
            query: Original search query.
            results: Search results to assess.
            domain: Domain classification.
            initial_confidence: Model confidence before retrieval.

        Returns:
            Tuple of (RetrievalQuality, quality_score).
        """
        if not results:
            return RetrievalQuality.INSUFFICIENT, 0.0

        # 1. Result count score
        count_score = min(len(results) / 5.0, 1.0)

        # 2. Relevance score (average)
        relevance_scores = [r.relevance_score for r in results if r.relevance_score > 0]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        # 3. Query coverage score
        query_terms = set(query.lower().split())
        covered_terms = set()
        for result in results:
            text = f"{result.title} {result.snippet}".lower()
            for term in query_terms:
                if term in text:
                    covered_terms.add(term)
        coverage_score = len(covered_terms) / len(query_terms) if query_terms else 0.0

        # 4. Content richness score
        rich_results = sum(1 for r in results if len(r.snippet) > 50)
        richness_score = rich_results / len(results) if results else 0.0

        # Composite score
        quality_score = (
            count_score * 0.2
            + avg_relevance * 0.3
            + coverage_score * 0.3
            + richness_score * 0.2
        )

        # Map score to quality level
        if quality_score >= 0.7:
            quality = RetrievalQuality.SUFFICIENT
        elif quality_score >= 0.4:
            quality = RetrievalQuality.PARTIAL
        elif quality_score >= 0.2:
            quality = RetrievalQuality.INSUFFICIENT
        else:
            quality = RetrievalQuality.IRRELEVANT

        return quality, quality_score

    def _refine_query(
        self,
        original_query: str,
        current_query: str,
        results: List[SearchResult],
        quality: RetrievalQuality,
        domain: Optional[str],
    ) -> Tuple[str, str]:
        """Refine the search query based on retrieval quality.

        Selects and applies a refinement strategy based on why
        the current results are insufficient.

        Args:
            original_query: The original user query.
            current_query: The current search query.
            results: Results from the current query.
            quality: Quality assessment of current results.
            domain: Domain classification.

        Returns:
            Tuple of (refined_query, strategy_name).
        """
        if quality == RetrievalQuality.IRRELEVANT:
            # Results don't match at all — try broadening
            return self._broaden_query(original_query, domain), self.REFINE_BROADEN

        if quality == RetrievalQuality.INSUFFICIENT:
            # Some results but not enough — try adding context from results
            return self._add_context_query(original_query, results, domain), self.REFINE_ADD_CONTEXT

        if quality == RetrievalQuality.PARTIAL:
            # Partial results — try decomposing or narrowing
            if self.enable_decomposition and len(original_query.split()) > 6:
                return self._decompose_query(original_query, domain), self.REFINE_DECOMPOSE
            return self._narrow_query(original_query, results, domain), self.REFINE_NARROW

        # Default: rephrase
        return self._rephrase_query(original_query, domain), self.REFINE_REPHRASE

    def _add_context_query(
        self,
        query: str,
        results: List[SearchResult],
        domain: Optional[str],
    ) -> str:
        """Add context terms from partial results to the query.

        Extracts key terms from the snippets of partial results
        and adds them to the query for better targeting.
        """
        # Extract significant terms from result snippets
        significant_terms = set()
        for result in results[:3]:
            words = result.snippet.lower().split()
            for word in words:
                if len(word) > 4 and word.isalpha():
                    significant_terms.add(word)

        # Pick the top 2-3 most relevant terms not already in query
        query_lower = query.lower()
        new_terms = [t for t in significant_terms if t not in query_lower][:2]

        if new_terms:
            return f"{query} {' '.join(new_terms)}"

        # Fallback: add domain context
        if domain:
            return f"{query} {domain} explained"
        return f"{query} guide tutorial"

    def _broaden_query(self, query: str, domain: Optional[str]) -> str:
        """Broaden a query that returned irrelevant results.

        Removes specific terms and adds more general ones.
        """
        words = query.split()
        if len(words) > 4:
            # Keep only the most important words (first few)
            broadened = " ".join(words[:4])
            if domain:
                broadened += f" {domain}"
            return broadened

        # Add general terms
        return f"{query} overview introduction"

    def _narrow_query(
        self,
        query: str,
        results: List[SearchResult],
        domain: Optional[str],
    ) -> str:
        """Narrow a query that returned too many partial results.

        Adds specific terms from the best results to narrow the focus.
        """
        # Find the most relevant result
        if results:
            best = max(results, key=lambda r: r.relevance_score)
            if best.title:
                # Add key terms from the best result's title
                title_words = [w for w in best.title.split() if len(w) > 3][:2]
                if title_words:
                    return f"{query} {' '.join(title_words)}"

        if domain:
            return f"{query} {domain} specific implementation"
        return f"{query} detailed explanation"

    def _rephrase_query(self, query: str, domain: Optional[str]) -> str:
        """Rephrase a query using alternative framing.

        Tries different query formulations to capture different
        aspects of the information need.
        """
        # Add "how to" or "what is" framing
        query_lower = query.lower()
        if not any(query_lower.startswith(p) for p in ["how to", "what is", "explain", "why"]):
            return f"what is {query}"

        # Reverse the framing
        if query_lower.startswith("how to"):
            return query.replace("how to", "guide for", 1)
        if query_lower.startswith("what is"):
            return query.replace("what is", "explain", 1)

        return f"{query} explained"

    def _decompose_query(
        self,
        query: str,
        domain: Optional[str],
    ) -> str:
        """Decompose a complex query into a focused sub-query.

        Instead of searching for the entire complex query,
        search for the most important sub-component.
        """
        # Split on conjunctions
        conjunctions = [" and ", ", then ", " then ", " followed by "]
        for conj in conjunctions:
            if conj in query.lower():
                parts = query.lower().split(conj)
                # Use the first (primary) part
                return parts[0].strip()

        # If no conjunctions, focus on the core terms
        words = query.split()
        if len(words) > 4:
            return " ".join(words[:4])

        return query

    def _synthesize_results(
        self,
        all_results: List[SearchResult],
        query: str,
    ) -> List[SearchResult]:
        """Synthesize results from multiple retrieval rounds.

        Deduplicates, re-ranks, and merges results from all rounds
        into a single coherent result set.
        """
        # Deduplicate by URL
        seen_urls: set = set()
        unique_results: List[SearchResult] = []

        for result in all_results:
            url_key = result.url.split("?")[0]  # Ignore query params for dedup
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique_results.append(result)
            else:
                # Merge: update relevance score to max
                for existing in unique_results:
                    existing_url = existing.url.split("?")[0]
                    if existing_url == url_key:
                        existing.relevance_score = max(
                            existing.relevance_score, result.relevance_score
                        )
                        # Keep the richer snippet
                        if len(result.snippet) > len(existing.snippet):
                            existing.snippet = result.snippet
                        break

        # Re-rank by relevance
        unique_results.sort(key=lambda r: r.relevance_score, reverse=True)

        return unique_results
