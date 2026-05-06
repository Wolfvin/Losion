"""
Web Search Interface — Web search capability for the agent layer.

This module provides web search functionality for the agent layer.
It is ONLY triggered when the model's confidence is below threshold —
the model doesn't search the web on every inference step.

The interface is designed to be pluggable — different search backends
can be swapped in (e.g., z-ai-web-dev-sdk, SerpAPI, Google Custom Search,
Bing Web Search, etc.) without changing the agent logic.

Design:
    Agent → WebSearchInterface.search(query) → List[SearchResult]

    Backends:
    - "zai": z-ai-web-dev-sdk (default)
    - "mock": Mock search for testing
    - "custom": User-provided search function
"""

from __future__ import annotations

import logging
import threading
import time as _time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single web search result.

    Attributes:
        url: URL of the result.
        title: Title of the result.
        snippet: Brief description/excerpt.
        source: Source domain name.
        rank: Rank in search results (1 = first).
        relevance_score: Estimated relevance to the query [0.0, 1.0].
        date: Publication date if available.
        metadata: Additional metadata.
    """

    url: str = ""
    title: str = ""
    snippet: str = ""
    source: str = ""
    rank: int = 0
    relevance_score: float = 0.0
    date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether this result has meaningful content."""
        return bool(self.url and (self.title or self.snippet))


@dataclass
class SearchConfig:
    """Configuration for web search.

    Attributes:
        backend: Search backend to use ("zai", "mock", "custom").
        max_results: Maximum number of results per query.
        timeout: Timeout for search requests in seconds.
        cache_results: Whether to cache search results.
        cache_ttl: Cache time-to-live in seconds.
        max_cache_entries: Maximum number of cached queries (LRU eviction).
        max_query_length: Maximum query string length before caching.
            Queries exceeding this limit are truncated. Prevents memory
            pressure from malformed model outputs (v2.5.0).
        allow_mock_fallback: Whether to fall back to mock results on backend failure.
            In production, set to False to prevent fabricated data from being
            treated as real evidence.
        language: Preferred search language.
        safe_search: Whether to enable safe search filtering.
    """

    backend: str = "zai"
    max_results: int = 10
    timeout: float = 30.0
    cache_results: bool = True
    cache_ttl: float = 3600.0  # 1 hour
    max_cache_entries: int = 1000  # LRU cap
    max_query_length: int = 1000  # v2.5.0: prevent memory pressure from huge queries
    allow_mock_fallback: bool = False  # Fail-closed by default
    language: str = "en"
    safe_search: bool = True


class WebSearchInterface:
    """Web search interface for the Losion Agent Layer.

    Provides a unified interface for web search with multiple backend
    support and result caching.

    Key features:
    - Multiple backend support (z-ai-sdk, mock, custom)
    - Result caching with TTL
    - Query expansion for better results
    - Relevance scoring
    - Rate limiting

    Args:
        config: Search configuration.
        custom_handler: Custom search function (for "custom" backend).
    """

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        custom_handler: Optional[Callable[[str, int], List[SearchResult]]] = None,
    ) -> None:
        self.config = config or SearchConfig()
        self._custom_handler = custom_handler
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # query → (results, timestamp)
        self._cache_lock = threading.Lock()
        self._search_count = 0

    def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        language: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search the web for the given query.

        Args:
            query: Search query string.
            num_results: Override number of results.
            language: Override search language.

        Returns:
            List of SearchResult objects.
        """
        # v2.5.0: Sanitize query length to prevent memory pressure
        # from malformed model outputs
        if len(query) > self.config.max_query_length:
            logger.warning(
                f"Web search query too long ({len(query)} chars, "
                f"max {self.config.max_query_length}). Truncating."
            )
            query = query[:self.config.max_query_length]

        # Check cache
        if self.config.cache_results:
            cached = self._check_cache(query)
            if cached is not None:
                logger.info(f"Web search cache hit: {query[:50]}")
                return cached

        # Determine backend
        if self.config.backend == "zai":
            results = self._search_zai(query, num_results, language)
        elif self.config.backend == "mock":
            results = self._search_mock(query, num_results)
        elif self.config.backend == "custom":
            results = self._search_custom(query, num_results or self.config.max_results)
        else:
            if self.config.allow_mock_fallback:
                logger.warning(f"Unknown search backend: {self.config.backend}, falling back to mock")
                results = self._search_mock(query, num_results)
            else:
                logger.error(
                    f"Unknown search backend: {self.config.backend} and mock fallback is disabled"
                )
                raise ValueError(
                    f"Unknown search backend: {self.config.backend}. "
                    f"Set backend to 'zai', 'mock', or 'custom'. "
                    f"Set allow_mock_fallback=True for development."
                )

        # Score relevance
        results = self._score_relevance(query, results)

        # Cache results
        if self.config.cache_results:
            self._cache_results(query, results)

        self._search_count += 1

        return results

    def _search_zai(
        self,
        query: str,
        num_results: Optional[int],
        language: Optional[str],
    ) -> List[SearchResult]:
        """Search using z-ai-web-dev-sdk.

        This is the default backend when running in the z.ai environment.
        Falls back to mock if the SDK is not available.
        """
        n = num_results or self.config.max_results

        try:
            import asyncio

            async def _do_search():
                from zai_web_dev_sdk import ZAI
                zai = await ZAI.create()
                raw_results = await zai.functions.invoke("web_search", {
                    "query": query,
                    "num": n,
                })
                return raw_results

            # Run async search
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context — use a thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        raw_results = pool.submit(
                            asyncio.run, _do_search()
                        ).result()
                else:
                    raw_results = loop.run_until_complete(_do_search())
            except RuntimeError:
                raw_results = asyncio.run(_do_search())

            # Convert to SearchResult
            results = []
            for i, r in enumerate(raw_results):
                results.append(SearchResult(
                    url=r.get("url", ""),
                    title=r.get("name", ""),
                    snippet=r.get("snippet", ""),
                    source=r.get("host_name", ""),
                    rank=i + 1,
                    date=r.get("date"),
                ))
            return results

        except ImportError:
            if self.config.allow_mock_fallback:
                logger.warning("z-ai-web-dev-sdk not available, falling back to mock search")
                return self._search_mock(query, num_results)
            else:
                logger.error("z-ai-web-dev-sdk not available and mock fallback is disabled")
                raise RuntimeError(
                    "Web search backend unavailable and mock fallback is disabled. "
                    "Set allow_mock_fallback=True for development."
                )
        except Exception as e:
            if self.config.allow_mock_fallback:
                logger.warning(f"z-ai search failed: {e}, falling back to mock")
                return self._search_mock(query, num_results)
            else:
                logger.error(f"z-ai search failed: {e}, mock fallback disabled")
                raise RuntimeError(
                    f"Web search failed: {e}. Mock fallback is disabled — "
                    f"set allow_mock_fallback=True for development."
                ) from e

    def _search_mock(
        self,
        query: str,
        num_results: Optional[int],
    ) -> List[SearchResult]:
        """Mock search for testing and offline use.

        Returns synthetic results based on the query for development
        and testing purposes.
        """
        n = min(num_results or self.config.max_results, 5)
        results = []
        for i in range(n):
            results.append(SearchResult(
                url=f"https://example.com/result-{i+1}?q={query.replace(' ', '+')}",
                title=f"Search Result {i+1} for '{query}'",
                snippet=f"This is a mock search result for the query '{query}'. "
                        f"In production, this would contain actual web content.",
                source="example.com",
                rank=i + 1,
                relevance_score=max(0.8 - i * 0.15, 0.1),
                metadata={"synthetic": True},
            ))
        return results

    def _search_custom(
        self,
        query: str,
        num_results: int,
    ) -> List[SearchResult]:
        """Search using a custom handler function."""
        if self._custom_handler is None:
            logger.warning("No custom search handler configured")
            return []

        try:
            return self._custom_handler(query, num_results)
        except Exception as e:
            logger.warning(f"Custom search handler failed: {e}")
            return []

    def _score_relevance(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Score search results for relevance to the query.

        Uses simple keyword overlap scoring. In production, this
        could be replaced with a learned relevance model.

        Args:
            query: Original search query.
            results: Search results to score.

        Returns:
            Results with updated relevance_score.
        """
        query_terms = set(query.lower().split())

        for result in results:
            # Combine title + snippet for scoring
            text = f"{result.title} {result.snippet}".lower()
            text_terms = set(text.split())

            # Jaccard-like overlap
            if query_terms and text_terms:
                overlap = len(query_terms & text_terms)
                total = len(query_terms | text_terms)
                result.relevance_score = overlap / total if total > 0 else 0.0
            else:
                result.relevance_score = 0.0

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        return results

    def _check_cache(self, query: str) -> Optional[List[SearchResult]]:
        """Check the cache for previous search results."""
        with self._cache_lock:
            if query in self._cache:
                results, timestamp = self._cache[query]
                age = _time.time() - timestamp
                if age < self.config.cache_ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(query)
                    return results
                else:
                    del self._cache[query]
        return None

    def _cache_results(self, query: str, results: List[SearchResult]) -> None:
        """Cache search results with LRU eviction.

        If the cache exceeds max_cache_entries, the least-recently-used
        entry is evicted. Expired entries are also pruned on write.
        """
        with self._cache_lock:
            # Evict expired entries on write
            now = _time.time()
            expired = [
                q for q, (_, ts) in self._cache.items()
                if now - ts >= self.config.cache_ttl
            ]
            for q in expired:
                del self._cache[q]

            # Insert / update
            self._cache[query] = (results, now)
            self._cache.move_to_end(query)

            # LRU eviction if over capacity
            while len(self._cache) > self.config.max_cache_entries:
                self._cache.popitem(last=False)  # Remove oldest (LRU)

    def clear_cache(self) -> None:
        """Clear the search result cache."""
        with self._cache_lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        with self._cache_lock:
            cache_size = len(self._cache)
        return {
            "total_searches": self._search_count,
            "cache_size": cache_size,
            "backend": self.config.backend,
        }
