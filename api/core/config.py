from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    cors_origins: list[str] = ['http://localhost:5173', 'http://localhost:3000']
    db_path: Path = Path(__file__).parent.parent.parent / 'db' / 'mtg.db'
    card_data_dir: Path = Path(__file__).parent.parent.parent / 'card data'

    class Config:
        env_file = '.env'


settings = Settings()
