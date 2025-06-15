import hashlib
import os
import re
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Set, Union

import pandas as pd
from openai import AsyncOpenAI
from pymongo import UpdateOne

from alligator.database import DatabaseAccessMixin
from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher
from alligator.log import get_logger
from alligator.mongo import MongoCache, MongoWrapper
from alligator.types import Candidate, Entity, RowData
from alligator.utils import ColumnHelper, clean_str


class RowBatchProcessor(DatabaseAccessMixin):
    """
    Extracted logic for process_rows_batch (and associated scoring helpers).
    Takes the Alligator instance so we can reference .mongo_wrapper, .feature, etc.
    """

    def __init__(
        self,
        dataset_name: str,
        table_name: str,
        feature: Feature,
        candidate_fetcher: CandidateFetcher,
        object_fetcher: ObjectFetcher | None = None,
        literal_fetcher: LiteralFetcher | None = None,
        max_candidates_in_result: int = 5,
        fuzzy_retry: bool = False,
        column_types: Mapping[str, Union[str, List[str]]] | None = None,
        enable_llm_filtering: bool = False,
        llm_model: str = "anthropic/claude-3.5-sonnet",
        openrouter_api_key: Optional[str] = None,
        openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        llm_top_k: int = 5,
        llm_cache_size: int = 1000,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.feature = feature
        self.candidate_fetcher = candidate_fetcher
        self.object_fetcher = object_fetcher
        self.literal_fetcher = literal_fetcher
        self.max_candidates_in_result = max_candidates_in_result
        self.fuzzy_retry = fuzzy_retry

        # LLM filtering configuration
        self.enable_llm_filtering = enable_llm_filtering
        self.llm_model = llm_model
        self._openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self._openrouter_api_url = openrouter_api_url
        self.llm_top_k = llm_top_k
        self.llm_cache_size = llm_cache_size

        # Process column_types to ensure they are all lists
        self.column_types = {}
        if column_types:
            for col, types_value in column_types.items():
                if isinstance(types_value, str):
                    self.column_types[col] = [types_value]
                elif isinstance(types_value, list):
                    self.column_types[col] = types_value
                else:
                    self.column_types[col] = []
        self._db_name = kwargs.get("db_name", "alligator_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://gator-mongodb:27017")
        self.input_collection = kwargs.get("input_collection", "input_data")
        self.candidate_collection = kwargs.get("candidate_collection", "candidates")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)
        self.logger = get_logger("processors")

        # Initialize LLM filter if enabled
        self.llm_filter = None
        if self.enable_llm_filtering and self._openrouter_api_key:
            try:
                self.llm_filter = LLMCandidateFilter(
                    api_key=self._openrouter_api_key,
                    api_url=self._openrouter_api_url,
                    model=self.llm_model,
                    top_k=self.llm_top_k,
                    cache_size=self.llm_cache_size,
                    mongo_uri=self._mongo_uri,
                    db_name=self._db_name,
                )
                self.logger.info("LLM candidate filtering enabled with MongoDB caching")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM filter: {e}")
                self.llm_filter = None
        elif self.enable_llm_filtering and not self._openrouter_api_key:
            self.logger.warning("LLM filtering enabled but no OpenRouter API key provided")

    async def process_rows_batch(self, docs):
        """
        Orchestrates the overall flow:
          1) Extract entities and row data
          2) Fetch and enhance candidates
          3) Process and save results
        """
        try:
            # 1) Extract entities and row data
            entities, row_data_list = self._extract_entities(docs)

            # 2) Fetch initial candidates in one batch
            candidates = await self._fetch_all_candidates(entities)

            # 3) Process each row and update DB
            await self._process_rows(row_data_list, candidates)

        except Exception:
            self.mongo_wrapper.log_to_db(
                "ERROR", "Error processing batch of rows", traceback.format_exc()
            )

    def _extract_entities(self, docs) -> tuple[List[Entity], List[RowData]]:
        """
        Extract entities and row data from documents.
        """
        entities = []
        row_data_list = []

        for doc in docs:
            row = doc["data"]
            cleaned_row = [clean_str(str(cell)) for cell in row]
            ne_columns = doc["classified_columns"].get("NE", {})
            lit_columns = doc["classified_columns"].get("LIT", {})
            context_columns = doc.get("context_columns", [])
            correct_qids = doc.get("correct_qids", {})
            row_index = doc.get("row_id")

            # Create row data
            row_data = RowData(
                doc_id=doc["_id"],
                row=cleaned_row,
                ne_columns=ne_columns,
                lit_columns=lit_columns,
                context_columns=context_columns,
                correct_qids=correct_qids,
                row_index=row_index,
            )
            row_data_list.append(row_data)

            # Extract entities from this row
            for col_idx, ner_type in ne_columns.items():
                normalized_col = ColumnHelper.normalize(col_idx)
                if not ColumnHelper.is_valid_index(normalized_col, len(row)):
                    continue

                cell_value = cleaned_row[ColumnHelper.to_int(normalized_col)]
                if not cell_value or pd.isna(cell_value):
                    continue

                qids = correct_qids.get(f"{row_index}-{normalized_col}", [])
                entity = Entity(
                    value=cell_value,
                    row_index=row_index,
                    col_index=normalized_col,
                    correct_qids=qids,
                    fuzzy=False,
                )
                entities.append(entity)

        return entities, row_data_list

    async def _fetch_all_candidates(self, entities: List[Entity]) -> Dict[str, List[Candidate]]:
        """
        Fetch candidates for all entities, with fuzzy retry for poor results.
        Now optimized to fetch only distinct mentions per batch.
        """
        fetch_groups = set()
        for entity in entities:
            qids_key = tuple(sorted(entity.correct_qids)) if entity.correct_qids else ()
            # Get column types for this entity's column
            column_types = self.column_types.get(entity.col_index, [])
            types_key = tuple(sorted(column_types))
            fetch_key = (entity.value, entity.fuzzy, qids_key, types_key)
            fetch_groups.add(fetch_key)

        unique_entities = []
        unique_fuzzies = []
        unique_qids = []
        unique_types = []
        for value, fuzzy, qids_tuple, types_tuple in fetch_groups:
            unique_entities.append(value)
            unique_fuzzies.append(fuzzy)
            unique_qids.append(list(qids_tuple))
            unique_types.append(list(types_tuple))

        self.logger.info(
            f"Fetching candidates for {len(unique_entities)} distinct mentions "
            f"(from {len(entities)} total entities)"
        )

        initial_results = await self.candidate_fetcher.fetch_candidates_batch(
            entities=unique_entities,
            fuzzies=unique_fuzzies,
            qids=unique_qids,
            types=unique_types,
        )

        retry_fetch_groups = set()
        for value, fuzzy, qids_tuple, types_tuple in fetch_groups:
            retrieved_candidates = initial_results.get(value, [])
            if self.fuzzy_retry and not fuzzy and len(retrieved_candidates) < 1:
                retry_key = (value, True, qids_tuple, types_tuple)  # fuzzy=True for retry
                retry_fetch_groups.add(retry_key)

        if retry_fetch_groups:
            retry_entities = []
            retry_fuzzies = []
            retry_qids = []
            retry_types = []
            for value, fuzzy, qids_tuple, types_tuple in retry_fetch_groups:
                retry_entities.append(value)
                retry_fuzzies.append(fuzzy)
                retry_qids.append(list(qids_tuple))
                retry_types.append(list(types_tuple))

            self.logger.info(f"Performing fuzzy retry for {len(retry_entities)} distinct mentions")

            retry_results = await self.candidate_fetcher.fetch_candidates_batch(
                entities=retry_entities,
                fuzzies=retry_fuzzies,
                qids=retry_qids,
                types=retry_types,
            )

            for value in retry_entities:
                if value in retry_results:
                    initial_results[value] = retry_results[value]

        # Convert to Candidate objects (reuse results for all entities with same mention)
        candidates: Dict[str, List[Candidate]] = defaultdict(list)
        for mention, candidates_list in initial_results.items():
            for candidate in candidates_list:
                candidate_dict: Dict[str, Any] = {"features": {}}
                for key, value in candidate.items():
                    if key in self.feature.selected_features:
                        candidate_dict["features"][key] = value
                    else:
                        candidate_dict[key] = value
                candidates[mention].append(Candidate.from_dict(candidate_dict))

        return candidates

    async def _process_rows(
        self, row_data_list: List[RowData], candidates: Dict[str, List[Candidate]]
    ):
        bulk_cand = []
        bulk_input = []
        db = self.get_db()

        """Process each row and update database."""
        for row_data in row_data_list:
            entity_ids = set()
            candidates_by_col = {}
            row_str = " ".join(str(v) for v in row_data.row)
            for col_idx, ner_type in row_data.ne_columns.items():
                normalized_col = ColumnHelper.normalize(col_idx)
                if not ColumnHelper.is_valid_index(normalized_col, len(row_data.row)):
                    continue

                cell_value = row_data.row[ColumnHelper.to_int(normalized_col)]
                if not cell_value or pd.isna(cell_value):
                    continue

                mention_candidates = candidates.get(cell_value, [])
                if mention_candidates:
                    candidates_by_col[normalized_col] = mention_candidates
                    for cand in mention_candidates:
                        if cand.id:
                            entity_ids.add(cand.id)

                # Apply LLM filtering and marking if enabled
                if self.llm_filter and mention_candidates:
                    await self.llm_filter.filter_and_mark_candidates(
                        cell_value, mention_candidates, row_data.row
                    )

                # Process each entity in the row
                self._compute_features(row_str, mention_candidates, row_data.row)

            # Enhance with additional features if possible
            if self.object_fetcher and self.literal_fetcher:
                await self._enhance_with_lamapi_features(row_data, entity_ids, candidates_by_col)

            # Update the status in the input collection
            bulk_input.append(
                UpdateOne(
                    {"_id": row_data.doc_id},
                    {
                        "$set": {
                            "status": "DONE",
                            "rank_status": "TODO",
                            "rerank_status": "TODO",
                        }
                    },
                )
            )

            # Store candidates in separate normalized documents
            for col_id, col_candidates in candidates_by_col.items():
                # Create one document per column
                bulk_cand.append(
                    UpdateOne(
                        {
                            "row_id": str(row_data.row_index),
                            "col_id": str(col_id),
                            "owner_id": row_data.doc_id,
                        },
                        {"$set": {"candidates": [c.to_dict() for c in col_candidates]}},
                        upsert=True,
                    )
                )
        bulk_batch_size = 8192
        if bulk_cand:
            for i in range(0, len(bulk_cand), bulk_batch_size):
                db[self.candidate_collection].bulk_write(
                    bulk_cand[i : i + bulk_batch_size], ordered=False
                )

        if bulk_input:
            for i in range(0, len(bulk_input), bulk_batch_size):
                db[self.input_collection].bulk_write(
                    bulk_input[i : i + bulk_batch_size], ordered=False
                )

    def _compute_features(self, row_str: str, candidates: List[Candidate], row: List[str]):
        """Process entities by computing features. Feature computation
        is done in-place over the candidates."""

        self.feature.process_candidates(candidates, row_str, row)

    async def _enhance_with_lamapi_features(
        self,
        row_data: RowData,
        entity_ids: Set[str],
        candidates_by_col: Dict[str, List[Candidate]],
    ):
        """Enhance candidates with LAMAPI features."""

        # Fetch external data
        objects_data = None
        if self.object_fetcher:
            objects_data = await self.object_fetcher.fetch_objects(list(entity_ids))

        literals_data = None
        if self.literal_fetcher:
            literals_data = await self.literal_fetcher.fetch_literals(list(entity_ids))

        if objects_data is not None:
            self.feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        if literals_data is not None:
            self.feature.compute_entity_literal_relationships(
                candidates_by_col, row_data.lit_columns, row_data.row, literals_data
            )


class LLMCandidateFilter:
    """
    Async LLM-based candidate filtering with MongoDB caching.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        model: str = "anthropic/claude-3.5-sonnet",
        top_k: int = 5,
        cache_size: int = 1000,
        mongo_uri: str = "mongodb://gator-mongodb:27017/",
        db_name: str = "alligator_db",
        cache_ttl_seconds: int = 86400,  # 24 hours
    ):
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=api_url.replace("/chat/completions", "")
        )
        self.model = model
        self.top_k = top_k
        self.logger = get_logger("llm_filter")

        # Initialize MongoDB cache
        self._cache = MongoCache(
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name="llm_candidate_cache",
            ttl_seconds=cache_ttl_seconds,
            capped_size_bytes=cache_size * 1024,  # Convert to bytes (approximate)
            capped_max_docs=cache_size,
        )

    def _generate_cache_key(
        self, mention: str, candidates: List[Candidate], context: List[Any]
    ) -> str:
        """Generate a cache key for the LLM request."""
        # Create a deterministic key based on mention, candidate IDs, and context
        candidate_ids = sorted([c.id for c in candidates])
        context_str = "|".join([str(x) for x in context if x])
        key_data = f"{mention}:{':'.join(candidate_ids)}:{context_str}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache."""
        try:
            return self._cache.get(cache_key)
        except Exception as e:
            self.logger.warning(f"Error reading from cache: {e}")
            return None

    def _put_in_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Put result in cache."""
        try:
            self._cache.put(cache_key, result)
        except Exception as e:
            self.logger.warning(f"Error writing to cache: {e}")

    async def filter_and_mark_candidates(
        self, mention: str, candidates: List[Candidate], row_context: List[Any]
    ) -> List[Candidate]:
        """
        Filter candidates using LLM and mark top-k as llm_chosen.

        Args:
            mention: The text mention from the table cell
            candidates: List of candidate entities (modified in-place)
            row_context: The complete row data for context

        Returns:
            All candidates with llm_chosen field updated for top-k
        """
        if not candidates:
            return candidates

        if len(candidates) <= 1:
            # Single candidate - mark as chosen
            candidates[0].llm_chosen = True
            return candidates

        try:
            # Check cache first
            cache_key = self._generate_cache_key(mention, candidates, row_context)
            cached_result = self._get_from_cache(cache_key)

            if cached_result:
                self.logger.debug(f"Using cached LLM result for mention '{mention}'")
                chosen_indices = cached_result["chosen_indices"]
            else:
                # Make async LLM call
                chosen_indices = await self._call_llm(mention, candidates, row_context)

                # Cache the result
                self._put_in_cache(cache_key, {"chosen_indices": chosen_indices})

            # Mark chosen candidates
            for i, candidate in enumerate(candidates):
                candidate.llm_chosen = i in chosen_indices

            self.logger.debug(
                f"LLM marked {len(chosen_indices)}/{len(candidates)} candidates "
                f"as chosen for mention '{mention}'"
            )

        except Exception as e:
            self.logger.error(f"Error in LLM candidate filtering: {e}")
            # Conservative fallback - mark top candidates by score as chosen
            sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
            top_candidates = sorted_candidates[: self.top_k]
            for candidate in candidates:
                candidate.llm_chosen = candidate in top_candidates

        return candidates

    async def _call_llm(
        self, mention: str, candidates: List[Candidate], row_context: List[Any]
    ) -> List[int]:
        """Make async LLM API call to select top-k candidates."""
        context_str = " | ".join([str(cell) for cell in row_context if cell])

        # Format candidates for the prompt
        candidate_info = []
        for i, candidate in enumerate(candidates):
            types_str = ", ".join([t.get("name", t.get("id", "")) for t in candidate.types[:3]])
            candidate_info.append(
                f"{i+1}. {candidate.name} ({candidate.id})"
                + (f" - Types: {types_str}" if types_str else "")
                + (f" - {candidate.description}" if candidate.description else "")
            )

        candidates_text = "\n".join(candidate_info)

        prompt = f"""Given this table row context: {context_str}

        For the mention "{mention}", select the TOP {self.top_k} most plausible candidates from the list below.
        Consider the table context to identify the most relevant matches.

        Candidates:
        {candidates_text}

        Return only the numbers (1, 2, 3, etc.) of the TOP {self.top_k} most plausible candidates, separated by commas.
        Order them by plausibility (most plausible first).
        If there are fewer than {self.top_k} plausible candidates, return only the plausible ones."""  # noqa: E501

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
                timeout=30,
            )

            content = response.choices[0].message.content or ""
            numbers = [int(x) for x in re.findall(r"\b\d+\b", content)]

            # Convert to 0-based indices and limit to top_k
            chosen_indices = []
            for num in numbers[: self.top_k]:
                if 1 <= num <= len(candidates):
                    chosen_indices.append(num - 1)

            # If no valid candidates were selected, fallback to top candidates by score
            if not chosen_indices:
                self.logger.warning(
                    f"LLM returned no valid candidates for mention '{mention}', using score fallback"  # noqa: E501
                )
                sorted_indices = sorted(
                    range(len(candidates)), key=lambda i: candidates[i].score, reverse=True
                )
                chosen_indices = sorted_indices[: self.top_k]

            return chosen_indices

        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            # Fallback to top candidates by score
            sorted_indices = sorted(
                range(len(candidates)), key=lambda i: candidates[i].score, reverse=True
            )
            return sorted_indices[: self.top_k]
