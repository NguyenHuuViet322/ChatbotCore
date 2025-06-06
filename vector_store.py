import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings  # Corrected import for Embeddings

def load_and_process_documents(data_folder: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Loads text documents from a folder and splits them into chunks.

    Args:
        data_folder: The path to the folder containing text documents.
        chunk_size: The desired size of text chunks.
        chunk_overlap: The overlap between consecutive chunks.

    Returns:
        A list of Document objects representing the text chunks.
    """
    print(f"Loading documents from {data_folder}...")
    documents = []
    # Check if the data folder exists
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return [] # Return empty list if folder doesn't exist

    # List files in the data folder
    try:
        filenames = os.listdir(data_folder)
    except OSError as e:
        print(f"Error listing files in {data_folder}: {e}")
        return []

    if not filenames:
         print(f"No files found in {data_folder}.")
         return []

    for filename in filenames:
        if filename.endswith('.txt'):
            full_path = os.path.join(data_folder, filename)
            try:
                # Using 'with' statement ensures the file is closed properly
                with open(full_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={'source': full_path}
                        )
                    )
                # print(f"Successfully loaded {full_path}") # Optional: for debugging
            except Exception as e:
                print(f"Error reading file {full_path}: {e}")

    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print("No documents loaded.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks.")
    return all_splits

def get_vectorstore(splits: List[Document], embedding_model: Embeddings, persist_directory: str) -> Chroma:
    """
    Creates a new Chroma vector store from document splits or loads an existing one.

    Args:
        splits: A list of Document objects (chunks) to populate the vector store.
        embedding_model: The embedding function to use.
        persist_directory: The directory where the vector store will be saved/loaded from.

    Returns:
        An initialized Chroma vector store instance.
    """
    db_path = os.path.join(persist_directory, "chroma.sqlite3")
    needs_rebuild = not os.path.exists(db_path)

    if needs_rebuild:
        if not splits:
            raise ValueError("Cannot build vector store: No document splits provided.")

        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        os.makedirs(persist_directory, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
    else:
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
        except Exception as e:
            raise ValueError("Failed to load vector store. Rebuilding is required.") from e

    return vectorstore

# Example usage (optional, for testing this module directly)
if __name__ == "__main__":
    print("Running vector_store.py directly for testing...")

    # Need a dummy embedding model for this test
    class DummyEmbeddings(Embeddings):
         def embed_documents(self, texts: List[str]) -> List[List[float]]:
             # Return dummy embeddings (e.g., list of zeros)
             print(f"Embedding {len(texts)} documents with DummyEmbeddings")
             return [[0.0] * 10 for _ in texts]
         def embed_query(self, text: str) -> List[float]:
             print(f"Embedding query '{text}' with DummyEmbeddings")
             return [0.0] * 10

    dummy_embedding = DummyEmbeddings()

    # Create a dummy data folder and file for testing
    dummy_data_folder = "./test_data_vs"
    dummy_persist_dir = "./test_vectorstore_vs"
    os.makedirs(dummy_data_folder, exist_ok=True)
    with open(os.path.join(dummy_data_folder, "test.txt"), "w", encoding='utf-8') as f:
        f.write("This is a test document.\nIt has multiple sentences.\nAnother sentence to make it longer.")
    with open(os.path.join(dummy_data_folder, "another_test.txt"), "w", encoding='utf-8') as f:
        f.write("This is a second test document.\nIt talks about different things.")


    try:
        # Test loading and processing
        print("\n--- Testing load_and_process_documents ---")
        test_splits = load_and_process_documents(dummy_data_folder, chunk_size=30, chunk_overlap=5)
        print(f"Test splits generated: {len(test_splits)}")
        if test_splits:
            print(f"First test split: {test_splits[0].page_content}")

        # Test creating vector store
        print("\n--- Testing vector store creation ---")
        # Clean up previous test vectorstore if it exists
        if os.path.exists(dummy_persist_dir):
             shutil.rmtree(dummy_persist_dir)
             print(f"Cleaned up {dummy_persist_dir}")

        test_vectorstore = get_vectorstore(test_splits, dummy_embedding, dummy_persist_dir)
        print("Test vector store created.")

        # Test loading vector store
        print("\n--- Testing vector store loading ---")
        # First, delete the splits to ensure loading from disk
        test_splits_for_loading = [] # Pass empty list to simulate not rebuilding
        test_vectorstore_loaded = get_vectorstore(test_splits_for_loading, dummy_embedding, dummy_persist_dir)
        print("Test vector store loaded successfully.")

        # Add a simple search test (will return based on dummy embeddings)
        print("\n--- Performing a dummy search ---")
        retriever = test_vectorstore_loaded.as_retriever(search_kwargs={"k": 2})
        search_results = retriever.invoke("test query")
        print(f"Dummy search result count: {len(search_results)}")
        if search_results:
            print(f"First dummy search result content: {search_results[0].page_content}")
            print(f"Second dummy search result content: {search_results[1].page_content}")


    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")

    finally:
        # Clean up test files and folders
        print("\n--- Cleaning up test environment ---")
        if os.path.exists(dummy_data_folder):
            shutil.rmtree(dummy_data_folder)
            print(f"Removed {dummy_data_folder}")
        if os.path.exists(dummy_persist_dir):
            shutil.rmtree(dummy_persist_dir)
            print(f"Removed {dummy_persist_dir}")
        print("Test cleanup complete.")