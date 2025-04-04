from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

data_path = "./app/data"

documents = SimpleDirectoryReader(data_path).load_data()

chunk_size = 2048
chunk_overlap = 256


text_splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
Settings.text_splitter = text_splitter

# indexing
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[text_splitter],
    embed_model=embed_model,
    show_progress=True
)

index.storage_context.persist(persist_dir="storage")
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)