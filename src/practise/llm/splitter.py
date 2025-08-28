from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

def get_splitter(splitter_type: str):
    if splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    elif splitter_type == "character":
        return CharacterTextSplitter()
    else:
        raise ValueError("Invalid splitter type. Choose 'recursive' or 'character'.")
