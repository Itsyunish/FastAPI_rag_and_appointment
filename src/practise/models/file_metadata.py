from pydantic import BaseModel

class FileMetadata(BaseModel):
    filename: str
    file_type: str
    size_bytes: int
    num_documents: int
    num_chunks: int
    splitter: str
