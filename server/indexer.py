import os
import torch
from typing import Optional, Callable, Dict
from byaldi import RAGMultiModalModel
from server.converters import convert_docs_to_pdfs
from server.logger import get_logger
from server.config import default_settings

logger = get_logger(__name__)


def index_documents(
    file_to_index: str,
    index_name: str = "document_index",
    index_path: Optional[str] = None,
    indexer_model: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    add_to_existing: bool = False,
):
    """
    Index the given list of file paths using Byaldi.

    - If `add_to_existing=True`, we load an existing index (if it exists)
      and call `RAG.add_to_index(...)` for each file.
    - Otherwise, we create a fresh index using `RAG.index(...)`.

    `convert_docs_to_pdfs` is also used for doc/docx -> pdf conversion,
    but only for the file paths we are actually indexing (not an entire folder).
    """

    if not file_to_index:
        raise ValueError("No file paths provided to index_documents.")

    if indexer_model is None:
        indexer_model = default_settings['indexerModels'][0]

    if indexer_model not in default_settings['indexerModels']:
        raise ValueError(
            f"Invalid indexer model: {indexer_model}. "
            f"Supported models: {', '.join(default_settings['indexerModels'])}"
        )

    try:
        logger.info(
            f"Starting indexing with add_to_existing={add_to_existing}, "
            f"index_name={index_name}, index_path={index_path}"
        )

        # Convert doc/docx -> pdf for each file individually
        if file_to_index.lower().endswith((".doc", ".docx")):
            parent_dir = os.path.dirname(file_to_index)
            convert_docs_to_pdfs(parent_dir)
            # After conversion, the resulting pdf is just file_path with .pdf replaced
            # but let's try to find it:
            possible_pdf = os.path.splitext(file_to_index)[0] + ".pdf"
            if os.path.exists(possible_pdf):
                converted_paths.append(possible_pdf)
            else:
                raise ValueError(f"Converted PDF not found: {possible_pdf}")

        # Device selection
        device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
        logger.info(f"Using device: {device}")

        # Go to index folder if specified
        original_cwd = os.getcwd()
        if index_path:
            os.makedirs(index_path, exist_ok=True)
            os.chdir(index_path)

        RAG = None
        full_index_dir = os.path.join(index_path or ".", '.byaldi', index_name)

        try:
            if add_to_existing:
                # 1) Attempt to load existing index
                if os.path.exists(full_index_dir):
                    logger.info(f"Loading existing index from: {full_index_dir}")
                    RAG = RAGMultiModalModel.from_index(
                        index_path=full_index_dir,
                        device=device
                    )
                    if RAG is None:
                        raise ValueError(
                            f"Failed to load an existing index at {full_index_dir}"
                        )
                    logger.info("Successfully loaded existing RAG index. Adding new file(s)...")

                    RAG.add_to_index(
                            input_item=file_to_index,
                            store_collection_with_index=True
                        )
                else:
                    logger.warning(
                        f"No existing index found at {full_index_dir}. Creating a new index instead."
                    )
                    RAG = RAGMultiModalModel.from_pretrained(
                        indexer_model, device=device
                    )
                    logger.info("Creating brand-new RAG index from scratch with these file(s)...")
                    dummy_temp_dir = "_tmp_init_index"
                    os.makedirs(dummy_temp_dir, exist_ok=True)
                    RAG.index(
                        input_path=dummy_temp_dir,
                        index_name=index_name,
                        store_collection_with_index=True,
                        overwrite=True
                    )
                    # Now the empty index is ready, we can add each file
                    RAG.add_to_index(
                        input_item=file_to_index,
                        store_collection_with_index=True
                    )
            else:
                logger.info("Creating new index from scratch (add_to_existing=False).")
                RAG = RAGMultiModalModel.from_pretrained(indexer_model, device=device)
                if RAG is None:
                    raise ValueError(
                        f"Failed to initialize RAGMultiModalModel with model {indexer_model}"
                    )

                dummy_temp_dir = "_tmp_init_index"
                os.makedirs(dummy_temp_dir, exist_ok=True)
                RAG.index(
                    input_path=dummy_temp_dir,
                    index_name=index_name,
                    store_collection_with_index=True,
                    overwrite=True
                )
                RAG.add_to_index(
                    input_item=file_to_index,
                    store_collection_with_index=True
                )

            # If no exceptions, we are done
            if progress_callback:
                progress_callback({'status': 'completed'})
            logger.info("Indexing completed successfully.")

        finally:
            # restore original directory
            os.chdir(original_cwd)

    except Exception as e:
        logger.error(f"Error in document indexing: {str(e)}")
        if progress_callback:
            progress_callback({'status': 'failed', 'error': str(e)})
        raise