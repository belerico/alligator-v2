import multiprocessing as mp
import os
import time
import uuid
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from column_classifier import ColumnClassifier

from alligator import PROJECT_ROOT
from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher
from alligator.ml import MLWorker
from alligator.processors import RowBatchProcessor
from alligator.typing import ColType


class Alligator:
    """
    Alligator entity linking system with hidden MongoDB configuration.
    """

    _DEFAULT_MONGO_URI = "mongodb://gator-mongodb:27017/"  # Change this to a class-level default
    _DB_NAME = "alligator_db"
    _INPUT_COLLECTION = "input_data"
    _ERROR_LOG_COLLECTION = "error_logs"
    _CACHE_COLLECTION = "candidate_cache"

    def __init__(
        self,
        input_csv: str | Path | pd.DataFrame,
        output_csv: str | Path | None = None,
        dataset_name: str = None,
        table_name: str = None,
        columns_type: ColType | None = None,
        max_workers: Optional[int] = None,
        max_candidates_in_result: int = 5,
        entity_retrieval_endpoint: Optional[str] = None,
        entity_retrieval_token: Optional[str] = None,
        selected_features: Optional[List[str]] = None,
        candidate_retrieval_limit: int = 16,
        ranker_model_path: Optional[str] = None,
        reranker_model_path: Optional[str] = None,
        batch_size: int = 1024,
        ml_ranking_workers: int = 2,
        top_n_for_type_freq: int = 3,
        **kwargs,
    ) -> None:
        from alligator.mongo import MongoWrapper

        self.input_csv = input_csv
        self.output_csv = output_csv
        if self.output_csv is None and kwargs.get("save_output_to_csv", True):
            if isinstance(self.input_csv, pd.DataFrame):
                raise ValueError(
                    "An output name must be specified is the input is a `pd.Dataframe`"
                )
            self.output_csv = os.path.splitext(input_csv)[0] + "_output.csv"
        if dataset_name is None:
            dataset_name = uuid.uuid4().hex
        if table_name is None:
            table_name = os.path.basename(self.input_csv).split(".")[0]
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.columns_type = columns_type
        self.max_workers = max_workers or mp.cpu_count()
        self.max_candidates_in_result = max_candidates_in_result
        self.entity_retrieval_endpoint = entity_retrieval_endpoint
        self.entity_retrieval_token = entity_retrieval_token
        self.candidate_retrieval_limit = candidate_retrieval_limit
        self.ranker_model_path = ranker_model_path or os.path.join(
            PROJECT_ROOT, "alligator", "models", "ranker.h5"
        )
        self.reranker_model_path = reranker_model_path or os.path.join(
            PROJECT_ROOT, "alligator", "models", "reranker.h5"
        )
        self.batch_size = batch_size
        self.ml_ranking_workers = ml_ranking_workers
        self.top_n_for_type_freq = top_n_for_type_freq
        self._mongo_uri = kwargs.pop("mongo_uri", None) or self._DEFAULT_MONGO_URI
        self._save_output_to_csv = kwargs.pop("save_output_to_csv", True)
        self.mongo_wrapper = MongoWrapper(
            self._mongo_uri, self._DB_NAME, self._ERROR_LOG_COLLECTION
        )
        self.feature = Feature(
            dataset_name,
            table_name,
            top_n_for_type_freq=top_n_for_type_freq,
            features=selected_features,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
        )

        # Instantiate our helper objects
        self._candidate_fetcher = CandidateFetcher(
            self.entity_retrieval_endpoint,
            self.entity_retrieval_token,
            self.candidate_retrieval_limit,
            self.feature,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
            cache_collection=self._CACHE_COLLECTION,
        )
        self._row_processor = RowBatchProcessor(
            self._candidate_fetcher,
            self.max_candidates_in_result,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
        )

        # Create indexes
        self.mongo_wrapper.create_indexes()

    def get_db(self):
        """Get MongoDB database connection for current process"""
        from alligator.mongo import MongoConnectionManager

        client = MongoConnectionManager.get_client(self._mongo_uri)
        return client[self._DB_NAME]

    def close_mongo_connection(self):
        """Cleanup when instance is destroyed"""
        from alligator.mongo import MongoConnectionManager

        try:
            MongoConnectionManager.close_connection()
        except Exception:
            pass

    def claim_todo_batch(self, input_collection, batch_size=16):
        docs = []
        for _ in range(batch_size):
            doc = input_collection.find_one_and_update(
                {
                    "dataset_name": self.dataset_name,
                    "table_name": self.table_name,
                    "status": "TODO",
                },
                {"$set": {"status": "DOING"}},
            )
            if doc is None:
                break
            docs.append(doc)
        return docs

    def onboard_data(
        self,
        dataset_name: str = None,
        table_name: str = None,
        columns_type: ColType | None = None,
    ):
        """Efficiently load data into MongoDB using batched inserts."""
        start_time = time.perf_counter()

        # Get database connection
        db = self.get_db()
        input_collection = db["input_data"]

        # Step 1: Determine data source and extract sample for classification
        if isinstance(self.input_csv, pd.DataFrame):
            df = self.input_csv
            sample = df
            total_rows = len(df)
            is_csv_path = False
        else:
            sample = pd.read_csv(self.input_csv, nrows=1024)
            total_rows = "unknown"
            is_csv_path = True

        print(f"Onboarding {total_rows} rows for dataset '{dataset_name}', table '{table_name}'")

        # Step 2: Perform column classification
        if columns_type is None:
            classifier = ColumnClassifier(model_type="fast")
            classification_results = classifier.classify_multiple_tables([sample])
            table_classification = classification_results[0].get("table_1", {})

            ne_cols, lit_cols, ignored_cols = {}, {}, []
            NE_classifications = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}

            for idx, col in enumerate(sample.columns):
                col_result = table_classification.get(col, {})
                classification = col_result.get("classification", "UNKNOWN")
                if classification in NE_classifications:
                    ne_cols[str(idx)] = classification
                else:
                    lit_cols[str(idx)] = classification
        else:
            ne_cols = columns_type.get("NE", {})
            lit_cols = columns_type.get("LIT", {})
            ignored_cols = columns_type.get("IGNORED", [])

        all_recognized_cols = set(ne_cols.keys()) | set(lit_cols.keys())
        all_cols = set([str(i) for i in range(len(sample.columns))])
        if len(all_recognized_cols) != len(all_cols):
            ignored_cols.extend(list(all_cols - all_recognized_cols))
        ignored_cols = list(set(ignored_cols))
        context_cols = list(set([str(i) for i in range(len(sample.columns))]) - set(ignored_cols))

        # Step 3: Define a chunk generator function
        def get_chunks():
            """Generator that yields chunks of rows, handling both DF and CSV."""
            if is_csv_path:
                chunk_size = 2048
                row_count = 0
                for chunk in pd.read_csv(self.input_csv, chunksize=chunk_size):
                    yield chunk, row_count
                    row_count += len(chunk)
            else:
                chunk_size = 1024 if total_rows > 100000 else 2048 if total_rows > 10000 else 4096
                total_chunks = (total_rows + chunk_size - 1) // chunk_size
                for chunk_idx in range(total_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, total_rows)
                    yield df.iloc[chunk_start:chunk_end], chunk_start

        # Step 4: Process all chunks using the generator
        processed_rows = 0
        chunk_idx = 0

        for chunk, start_idx in get_chunks():
            chunk_idx += 1
            documents = []
            for i, (_, row) in enumerate(chunk.iterrows()):
                row_id = start_idx + i
                document = {
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": row_id,
                    "data": row.tolist(),
                    "classified_columns": {
                        "NE": ne_cols,
                        "LIT": lit_cols,
                        "IGNORED": ignored_cols,
                    },
                    "context_columns": context_cols,
                    "correct_qids": {},
                    "status": "TODO",
                }
                documents.append(document)

            if documents:
                try:
                    input_collection.insert_many(documents, ordered=False)
                    chunk_size = len(documents)
                    processed_rows += chunk_size
                    elapsed = time.perf_counter() - start_time
                    rows_per_second = processed_rows / elapsed if elapsed > 0 else 0

                    if is_csv_path:
                        print(
                            f"Chunk {chunk_idx}: "
                            f"Processed {chunk_size} rows (total: {processed_rows}) "
                            f"({rows_per_second:.1f} rows/sec)"
                        )
                    else:
                        chunk_start = start_idx + 1
                        chunk_end = start_idx + chunk_size
                        total_chunks = (total_rows + chunk_size - 1) // chunk_size
                        print(
                            f"Chunk {chunk_idx}/{total_chunks}: "
                            f"Onboarded rows {chunk_start}-{chunk_end} "
                            f"({rows_per_second:.1f} rows/sec)"
                        )
                except Exception as e:
                    print(f"Error inserting batch {chunk_idx}: {str(e)}")
                    if "duplicate key" not in str(e).lower():
                        raise

        total_time = time.perf_counter() - start_time
        print(f"Data onboarding complete for dataset '{dataset_name}' and table '{table_name}'")
        print(
            f"Onboarded {processed_rows} rows in {total_time:.2f} seconds "
            f"({processed_rows/total_time:.1f} rows/sec)"
        )

    def fetch_results(self):
        """Retrieves processed documents from MongoDB using memory-efficient streaming.

        For large tables, this avoids loading all results into memory at once.
        Instead, it processes documents in batches and writes directly to CSV.
        """
        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        # Determine if we need to write to CSV or return full results
        stream_to_csv = self._save_output_to_csv and isinstance(self.output_csv, (str, Path))

        # Get header information
        header = None
        if isinstance(self.input_csv, pd.DataFrame):
            header = self.input_csv.columns.tolist()
        elif isinstance(self.input_csv, str):
            header = pd.read_csv(self.input_csv, nrows=0).columns.tolist()
        num_cols = len(header)

        # Get first document to determine column count if header is still None
        sample_doc = input_collection.find_one(
            {"dataset_name": self.dataset_name, "table_name": self.table_name}
        )
        if not sample_doc:
            print("No documents found for the specified dataset and table.")
            return []

        if header is None:
            print("Could not extract header from input table, using generic column names.")
            header = [f"col_{i}" for i in range(num_cols)]

        # Process in batches with cursor
        batch_size = 1024  # Process 1024 documents at a time

        # Only fetch fields we actually need to reduce network transfer
        projection = {"data": 1, "el_results": 1, "classified_columns.NE": 1}
        cursor = input_collection.find(
            {"dataset_name": self.dataset_name, "table_name": self.table_name},
            projection=projection,
        ).batch_size(batch_size)

        # Count documents if streaming to CSV for progress reporting
        if stream_to_csv:
            total_docs = input_collection.count_documents(
                {"dataset_name": self.dataset_name, "table_name": self.table_name}
            )
            print(f"Streaming {total_docs} documents to CSV...")

        # Setup for handling results
        all_rows = [] if not stream_to_csv else None
        current_chunk = []
        processed_count = 0
        chunk_size = 256  # Size for chunked CSV writing

        # Process all documents
        for doc in cursor:
            # Process each document - this is the common code between both paths
            row_data = self._extract_row_data(doc, header)

            if stream_to_csv:
                # For CSV streaming, collect into chunk
                current_chunk.append(row_data)
                processed_count += 1

                # Write chunk when it reaches desired size
                if len(current_chunk) >= chunk_size:
                    self._write_csv_chunk(current_chunk, processed_count, chunk_size, total_docs)
                    current_chunk = []
            else:
                # For memory collection, just append to results
                all_rows.append(row_data)
                processed_count += 1

                # Report progress periodically
                if processed_count % batch_size == 0:
                    print(f"Processed {processed_count} rows...")

        # Handle any remaining rows for CSV streaming
        if stream_to_csv and current_chunk:
            self._write_csv_chunk(current_chunk, processed_count, chunk_size, total_docs)
            print(f"Results saved to '{self.output_csv}'. Total rows: {processed_count}")
            return []
        elif not stream_to_csv:
            print(f"Retrieved {processed_count} rows total")
            return all_rows
        else:
            return []

    def _extract_row_data(self, doc, header):
        """Extract row data from a MongoDB document.

        Encapsulates the common logic for formatting a row from a document.
        """
        # Create base row data with original values
        row_data = dict(zip(header, doc["data"]))
        el_results = doc.get("el_results", {})

        # Add entity linking results
        for col_idx, col_type in doc["classified_columns"].get("NE", {}).items():
            col_index = int(col_idx)
            col_header = header[col_index]

            id_field = f"{col_header}_id"
            name_field = f"{col_header}_name"
            desc_field = f"{col_header}_desc"
            score_field = f"{col_header}_score"

            # Get first candidate or empty placeholder
            candidate = el_results.get(col_idx, [{}])[0]

            row_data[id_field] = candidate.get("id", "")
            row_data[name_field] = candidate.get("name", "")
            row_data[desc_field] = candidate.get("description", "")
            row_data[score_field] = candidate.get("score", 0)

        return row_data

    def _write_csv_chunk(self, chunk, processed_count, chunk_size, total_docs):
        """Write a chunk of data to CSV.

        Encapsulates the CSV writing logic.
        """
        # Create DataFrame and append to CSV
        chunk_df = pd.DataFrame(chunk)

        # Use mode='a' (append) for all chunks after the first
        mode = "w" if processed_count <= len(chunk) else "a"
        # Only include header for the first chunk
        header_option = True if processed_count <= len(chunk) else False

        # Write chunk to CSV
        chunk_df.to_csv(self.output_csv, index=False, mode=mode, header=header_option)

        # Report progress
        print(f"Processed {processed_count}/{total_docs} rows...")

    def ml_worker(
        self,
        rank: int,
        stage: str = "rank",
        global_type_counts: Dict[Any, Counter] = None,
    ):
        """Unified wrapper function for ML workers"""
        worker = MLWorker(
            rank,
            table_name=self.table_name,
            dataset_name=self.dataset_name,
            stage=stage,
            error_log_collection_name=self._ERROR_LOG_COLLECTION,
            input_collection=self._INPUT_COLLECTION,
            model_path=self.ranker_model_path if stage == "rank" else self.reranker_model_path,
            batch_size=self.batch_size,
            max_candidates_in_result=self.max_candidates_in_result if stage == "rerank" else -1,
            top_n_for_type_freq=self.top_n_for_type_freq,
            features=self.feature.selected_features,
            mongo_uri=self._mongo_uri,
            db_name=self._DB_NAME,
        )
        return worker.run(global_type_counts=global_type_counts)

    def worker(self, rank: int):
        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]
        while True:
            todo_docs = self.claim_todo_batch(input_collection)
            if not todo_docs:
                print("No more tasks to process.")
                break
            tasks_by_table = {}
            for doc in todo_docs:
                dataset_name = doc["dataset_name"]
                table_name = doc["table_name"]
                tasks_by_table.setdefault((dataset_name, table_name), []).append(doc)
            for (dataset_name, table_name), docs in tasks_by_table.items():
                self._row_processor.process_rows_batch(docs, dataset_name, table_name)

    def run(self):
        self.onboard_data(self.dataset_name, self.table_name, columns_type=self.columns_type)

        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        total_rows = self.mongo_wrapper.count_documents(input_collection, {"status": "TODO"})
        print(f"Found {total_rows} tasks to process.")

        with mp.Pool(processes=self.max_workers) as pool:
            pool.map(self.worker, range(self.max_workers))

        with mp.Pool(processes=self.ml_ranking_workers) as pool:
            pool.map(
                partial(
                    self.ml_worker,
                    stage="rank",
                    global_type_counts={},
                ),
                range(self.ml_ranking_workers),
            )

        global_type_counts = self.feature.compute_global_type_frequencies(
            docs_to_process=0.7, random_sample=True
        )

        with mp.Pool(processes=self.ml_ranking_workers) as pool:
            pool.map(
                partial(
                    self.ml_worker,
                    stage="rerank",
                    global_type_counts=global_type_counts,
                ),
                range(self.ml_ranking_workers),
            )

        print("All tasks have been processed.")
        extracted_rows = self.fetch_results()
        return extracted_rows
