import os
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_api_key: SecretStr
    pinecone_api_key: SecretStr
    mailtrap_api_key: SecretStr
    index_host: SecretStr

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
# GOOGLE_API_KEY=settings.google_api_key.get_secret_value()
# MAILTRAP_API_KEY = settings.mailtrap_api_key.get_secret_value()
# PINECONE_API_KEY = settings.pinecone_api_key.get_secret_value()
# INDEX_HOST = settings.index_host.get_secret_value()


UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
