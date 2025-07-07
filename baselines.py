import os
import jsonlines
import random
import ast
import argparse
import torch
from tqdm import tqdm
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi
from mlx_lm import load, generate
from kopyt import Parser, node

argparser = argparse.ArgumentParser()
# Parameters for context collection strategy
argparser.add_argument("--stage", type=str, default="practice", help="Stage of the project")
argparser.add_argument("--lang", type=str, default="python", help="Language")
argparser.add_argument("--strategy", type=str, default="random", help="Context collection strategy")

# Parameters for context trimming
argparser.add_argument("--trim-prefix", action="store_true", help="Trim the prefix to 10 lines")
argparser.add_argument("--trim-suffix", action="store_true", help="Trim the suffix to 10 lines")

args = argparser.parse_args()

stage = args.stage
language = args.lang
strategy = args.strategy

if language == "python":
    extension = ".py"
elif language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

print(f"Running the {strategy} baseline for stage '{stage}'")

# token used to separate different files in the context
FILE_SEP_SYMBOL = "<|file_sep|>"
# format to compose context from a file
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"


def find_random_file(root_dir: str, min_lines: int = 10) -> str:
    """
    Select a random file:
        - in the given language
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files with given extension.
    :param min_lines: Minimum number of lines required in the file.
    :return: Selected random file or None if no files were found.
    """
    code_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            code_files.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass

    return random.choice(code_files) if code_files else None


def find_bm25_file(root_dir: str, prefix: str, suffix: str, min_lines: int = 10) -> str:
    """
    Select the file:
        - in the given language
        - with the highest BM25 score with the completion file
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files.
    :param prefix: Prefix of the completion file.
    :param suffix: Suffix of the completion file.
    :param min_lines: Minimum number of lines required in the file.
    :return:
    """

    def prepare_bm25_str(s: str) -> list[str]:
        return "".join(c if c.isalnum() else " " for c in s.lower()).split()

    corpus = []
    file_names = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            content = "\n".join(lines)
                            content = prepare_bm25_str(content)
                            corpus.append(content)
                            file_names.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass

    query = (prefix + " " + suffix).lower()
    query = prepare_bm25_str(query)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)
    best_idx = scores.argmax()

    return file_names[best_idx] if file_names else None


def hybrid_search_file(root_dir: str, prefix: str, suffix: str, min_lines: int = 10) -> str:
    """
    Select the file:
        - in the given language
        - with the highest hybrid score (BM25 + embeddings) with the completion file
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files.
    :param prefix: Prefix of the completion file.
    :param suffix: Suffix of the completion file.
    :param min_lines: Minimum number of lines required in the file.
    :return: Selected file path or None if no files were found.
    """

    file_contents = []
    file_names = []

    # Traverse files and collect content
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            content = "\n".join(lines)
                            file_contents.append(content)
                            file_names.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass

    if not file_contents:
        return None

    # Create hybrid retriever
    documents = [Document(page_content=content) for content in file_contents]
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 1  # We want top 1 result
    
    # Vector retriever with OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    # Ensemble retriever combining both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]  # BM25 weight: 0.4, Vector weight: 0.6
    )
    
    # Query with prefix and suffix
    query = prefix + " " + suffix
    results = ensemble_retriever.invoke(query)
    
    if results:
        # Find the index of the best result in our original list
        best_content = results[0].page_content
        best_idx = file_contents.index(best_content)
        return file_names[best_idx]
    
    return None


def hybrid_search_file_local(root_dir: str, prefix: str, suffix: str, min_lines: int = 10) -> str:
    """
    Select the file:
        - in the given language
        - with the highest hybrid score (BM25 + local embeddings) with the completion file
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files.
    :param prefix: Prefix of the completion file.
    :param suffix: Suffix of the completion file.
    :param min_lines: Minimum number of lines required in the file.
    :return: Selected file path or None if no files were found.
    """

    file_contents = []
    file_names = []

    # Traverse files and collect content
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            content = "\n".join(lines)
                            file_contents.append(content)
                            file_names.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass

    if not file_contents:
        return None

    # Create hybrid retriever
    documents = [Document(page_content=content) for content in file_contents]
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 1  # We want top 1 result
    
    # Vector retriever with local HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        # model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    # Ensemble retriever combining both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]  # BM25 weight: 0.4, Vector weight: 0.6
    )
    
    # Query with prefix and suffix
    query = prefix + " " + suffix
    results = ensemble_retriever.invoke(query)
    
    if results:
        # Find the index of the best result in our original list
        best_content = results[0].page_content
        best_idx = file_contents.index(best_content)
        return file_names[best_idx]
    
    return None


def find_random_recent_file(root_dir: str, recent_filenames: list[str], min_lines: int = 10) -> str:
    """
    Select the most recent file:
        - in the given language
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files.
    :param recent_filenames: List of recent files filenames.
    :param min_lines: Minimum number of lines required in the file.
    :return: Selected random file or None if no files were found.
    """
    code_files = []
    for filename in recent_filenames:
        if filename.endswith(extension):
            file_path = os.path.join(root_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= min_lines:
                        code_files.append(file_path)
            except Exception as e:
                # Optional: handle unreadable files
                # print(f"Could not read {file_path}: {e}")
                pass
    return random.choice(code_files) if code_files else None


def trim_prefix(prefix: str):
    prefix_lines = prefix.split("\n")
    if len(prefix_lines) > 10:
        prefix = "\n".join(prefix_lines[-10:])
    return prefix

def trim_suffix(suffix: str):
    suffix_lines = suffix.split("\n")
    if len(suffix_lines) > 10:
        suffix = "\n".join(suffix_lines[:10])
    return suffix

def chunk_code_and_store_embeddings(root_dir: str, vector_db_path: str, min_lines: int = 8, overlap_size: int = 2):
    """
    Chunk code using AST structure, compute embeddings, and store them in a vector database.

    :param root_dir: Directory to search for code files.
    :param vector_db_path: Path to store the vector database.
    :param min_lines: Minimum number of lines required in a chunk.
    :param overlap_size: Number of overlapping lines between chunks.
    :return: Tuple containing vector store, BM25 retriever, and list of documents.
    """
    documents = []

    # Traverse files and chunk code using AST
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    # Use syntax-aware chunking
                    if extension == ".py":
                        file_documents = chunk_code_with_ast(file_path, min_lines=min_lines, overlap_size=overlap_size)
                    else:
                        file_documents = chunk_code_with_kopyt(file_path, min_lines=min_lines, overlap_size=overlap_size)
                    documents.extend(file_documents)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue

    # Compute embeddings and store in vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'token': os.environ.get('HUGGINGFACE_TOKEN')}
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    torch.mps.empty_cache()
    vector_store.save_local(vector_db_path)

    # Compute BM25 scores
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_corpus = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(bm25_corpus)

    return vector_store, bm25, bm25_retriever, documents


def retrieve_top_k_chunks(prefix: str, suffix: str, vector_store: FAISS, bm25: BM25Okapi, bm25_retriever: BM25Retriever, documents: Document, top_k: int = 15):
    """
    Retrieve top-k code chunks using ensemble of embeddings and BM25 scores.

    :param prefix: Prefix of the code.
    :param suffix: Suffix of the code.
    :param vector_store: Vector database storing code embeddings.
    :param bm25: BM25 retriever for code chunks.
    :param code_chunks: List of code chunks.
    :param chunk_metadata: Metadata for code chunks.
    :param top_k: Number of top chunks to retrieve.
    :return: List of top-k code chunks with metadata.
    """
    print ("--------- Retrieving top-k code chunks -----------")
    bm25_retriever.k = top_k
    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            vector_store.as_retriever(search_kwargs={"k": top_k}),
            bm25_retriever
        ],
        weights=[0.8, 0.2]  # Adjusted weights for embeddings and BM25
    )

    # Get top-k results
    query = prefix + " " + suffix
    top_k_results = ensemble_retriever.invoke(query)

    # Ensure no duplicates in the results
    unique_results = []
    seen_contents = set()
    for result in top_k_results:
        if result.page_content not in seen_contents:
            unique_results.append(result)
            seen_contents.add(result.page_content)

    print(f"Vector store size: {vector_store.index.ntotal}")
    print(f"Top-k results from ensemble retriever: {len(top_k_results)}")
    print(f"Unique results after filtering: {len(unique_results)}")

    print ("--------- Retrieving top-k code chunks -----------")
    return unique_results



def generate_filler_summary_with_model(prefix: str, suffix: str) -> str:
    """
    Generate a natural language summary of the expected filler between prefix and suffix using mlx-lm.

    :param prefix: Prefix of the code.
    :param suffix: Suffix of the code.
    :return: Natural language summary of the expected filler.
    """
    torch.mps.empty_cache()
    # Load the model and tokenizer using mlx-lm
    model_name = "mlx-community/Qwen2.5-Coder-3B-Instruct-bf16"
    model, tokenizer = load(model_name, tokenizer_config={"eos_token": "<|im_end|>"})

    # Prepare the prompt for the model
    prompt = f"Summarize the prefix code and suffix code in a structure. Don't repeat the suffix or prefix code in summary. There is a missing piece of code after prefix and before suffix. Reason what could be the missing piece that ties up prefix with suffix to achieve a functionality:\nPrefix:\n{prefix}\nSuffix:\n{suffix}"
    messages = [
        {"role": "system", "content": "You are an AI coding assistant. Your job is to summarize the code. You task also is to reason about missing piece of code to fulfill a functionality."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate the summary
    response = generate(
        model,
        tokenizer,
        prompt=text,
        verbose=True,
        # top_p=0.8,
        # temp=0.7,
        # repetition_penalty=1.05,
        max_tokens=500  # Adjust max_tokens for the summary length
    )
    torch.mps.empty_cache()
    return response


def chunk_code_with_ast(file_path: str, min_lines: int = 8, overlap_size: int = 2):
    """
    Parse the code into AST, chunk it into functions, classes, and loops, and embed metadata into the code chunk.

    :param file_path: Path to the code file.
    :param min_lines: Minimum number of lines required in a chunk.
    :param overlap_size: Number of overlapping lines between chunks.
    :return: List of Document objects with metadata embedded in the code chunk.
    """
    documents = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            tree = ast.parse(code)

            # Traverse AST to identify logical units
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.For, ast.While, ast.If)):
                    start_line = node.lineno
                    end_line = max([child.lineno for child in ast.walk(node) if hasattr(child, 'lineno')], default=start_line)
                    chunk = "\n".join(code.splitlines()[start_line - 1:end_line])
                    if len(chunk.splitlines()) >= min_lines:
                        # Embed metadata directly into the code chunk
                        metadata = f"# Metadata: file_name={file_path.split('/')[-1]}, function_name={getattr(node, 'name', 'N/A')}, start_line={start_line}, end_line={end_line}\n"
                        chunk_with_metadata = metadata + chunk
                        documents.append(Document(page_content=chunk_with_metadata))

            # Add overlapping chunks for continuity
            for i in range(len(documents) - 1):
                overlap_chunk = "\n".join(documents[i].page_content.splitlines()[-overlap_size:] +
                                          documents[i + 1].page_content.splitlines()[:overlap_size])
                documents.append(Document(page_content=overlap_chunk))

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return documents


def walk_kopyt_tree(tree):
    """
    Recursively yield all nodes in the kopyt AST tree.
    """
    yield tree
    if hasattr(tree, 'children') and tree.children:
        for child in tree.children:
            yield from walk_kopyt_tree(child)


def chunk_code_with_kopyt(file_path: str, min_lines: int = 8, overlap_size: int = 2):
    """
    Parse Kotlin code using kopyt, chunk it into functions, classes, and loops, and embed metadata into the code chunk.

    :param file_path: Path to the Kotlin code file.
    :param min_lines: Minimum number of lines required in a chunk.
    :param overlap_size: Number of overlapping lines between chunks.
    :return: List of Document objects with metadata embedded in the code chunk.
    """
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            lines = code.splitlines()
            parser = Parser(code)
            tree = parser.parse()

            # Helper to get line numbers from node positions
            def get_line(pos):
                return code[:pos].count('\n') + 1

            # Traverse the AST and collect nodes of interest
            nodes = []
            for n in walk_kopyt_tree(tree):
                if hasattr(n, 'type') and n.type in ["functionDeclaration", "classDeclaration", "forExpression", "whileExpression"]:
                    start_line = get_line(n.start)
                    end_line = get_line(n.end)
                    name = getattr(n, "name", n.type)
                    # Try to get the signature if available
                    signature = ""
                    if hasattr(n, "children"):
                        for child in n.children:
                            if hasattr(child, 'type') and child.type == "functionValueParameters":
                                signature = code[child.start:child.end]
                    nodes.append((start_line, end_line, n.type, name, signature))

            # Sort nodes by start_line
            nodes.sort(key=lambda x: x[0])

            # Chunk and embed metadata
            for start_line, end_line, node_type, name, signature in nodes:
                chunk_lines = lines[start_line-1:end_line]
                if len(chunk_lines) >= min_lines:
                    metadata = (
                        f"// Metadata: file_name={file_path.split('/')[-1]}, "
                        f"element_type={node_type}, element_name={name}, "
                        f"signature={signature}, start_line={start_line}, end_line={end_line}\n"
                    )
                    chunk_with_metadata = metadata + "\n".join(chunk_lines)
                    documents.append(Document(page_content=chunk_with_metadata))

            # Add overlapping chunks for continuity
            for i in range(len(documents) - 1):
                overlap_chunk = "\n".join(
                    documents[i].page_content.splitlines()[-overlap_size:] +
                    documents[i + 1].page_content.splitlines()[:overlap_size]
                )
                documents.append(Document(page_content=overlap_chunk))

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return documents


# Path to the file with completion points
completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")

# Path to the file to store predictions
prediction_file_name = f"{language}-{stage}-{strategy}"
if args.trim_prefix:
    prediction_file_name += "-short-prefix"
if args.trim_suffix:
    prediction_file_name += "-short-suffix"
predictions_file = os.path.join("predictions", f"{prediction_file_name}.jsonl")

with jsonlines.open(completion_points_file, 'r') as reader:
    with jsonlines.open(predictions_file, 'w') as writer:
        processed_count = 0  # Initialize counter

        for datapoint in reader:
            processed_count += 1  # Increment counter
            print(f"Processing datapoint {processed_count}...")

            # Identify the repository storage for the datapoint
            repo_path = datapoint['repo'].replace("/", "__")
            repo_revision = datapoint['revision']
            root_directory = os.path.join("data", f"repositories-{language}-{stage}", f"{repo_path}-{repo_revision}")

            # Run the baseline strategy
            if strategy == "random":
                file_name = find_random_file(root_directory)
            elif strategy == "bm25":
                file_name = find_bm25_file(root_directory, datapoint['prefix'], datapoint['suffix'])
            elif strategy == "recent":
                recent_filenames = datapoint['modified']
                file_name = find_random_recent_file(root_directory, recent_filenames)
                # If no recent files match our filtering criteria, select a random file instead
                if file_name is None:
                    file_name = find_random_file(root_directory)
            elif strategy == "hybrid":
                file_name = hybrid_search_file_local(root_directory, datapoint['prefix'], datapoint['suffix'])
            elif strategy == "code-chunk":
                vector_db_path = os.path.join("data", "vector_db")
                vector_store, bm25, bm25_retriever, documents = chunk_code_and_store_embeddings(root_directory, vector_db_path)
                top_k_chunks = retrieve_top_k_chunks(datapoint['prefix'], datapoint['suffix'], vector_store, bm25, bm25_retriever, documents, top_k=30)

                context_parts = []
                for chunk in top_k_chunks:
                    try:
                        context_part = FILE_COMPOSE_FORMAT.format(
                            file_sep=FILE_SEP_SYMBOL,
                            # file_name=chunk.metadata['file_name'],
                            file_name=chunk.metadata.get('file_name', 'unknown_file'),
                            file_content=chunk.page_content
                        )
                        context_parts.append(context_part)
                    except Exception as e:
                        print(f"Skipping chunk due to error: {e}")
                        continue

                # Generate natural language summary using the model
                filler_summary = generate_filler_summary_with_model(datapoint['prefix'], datapoint['suffix'])

                # Add summary to context
                context_parts.insert(0, f"Natural Language Summary: {filler_summary}")

                context = "\n".join(context_parts)
                submission = {"context": context}
                print(f"Top-k chunks retrieved and merged: {len(top_k_chunks)}")
                writer.write(submission)
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Compose the context from the selected file
            file_content = open(file_name, 'r', encoding='utf-8').read()
            clean_file_name = file_name[len(root_directory) + 1:]
            context = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                 file_content=file_content)

            submission = {"context": context}
            # Write the result to the prediction file
            print(f"Picked file: {clean_file_name}")
            if args.trim_prefix:
                submission["prefix"] = trim_prefix(datapoint["prefix"])
            if args.trim_suffix:
                submission["suffix"] = trim_suffix(datapoint["suffix"])
            writer.write(submission)