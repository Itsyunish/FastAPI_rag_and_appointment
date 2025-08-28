from pathlib import Path

def ensure_upload_dir(upload_dir: str):
    Path(upload_dir).mkdir(exist_ok=True)

def format_size_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024*1024):.2f} MB"
