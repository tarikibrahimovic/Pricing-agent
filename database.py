import os
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import logging

# Postavka loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
if not DATABASE_URL:
    raise Exception("SUPABASE_DATABASE_URL mora biti postavljen u .env fajlu")

Base = declarative_base()

class Product(Base):
    __tablename__ = "Product"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    count = Column(Integer, default=0)
    reward = Column(Float, default=1.0)
    
    def to_dict(self) -> dict:
        """Konvertuj model u dictionary za JSON serijalizaciju"""
        return {
            "id": self.id,
            "name": self.name,
            "count": self.count,
            "reward": self.reward
        }
    
    def __repr__(self) -> str:
        return f"<Product(id={self.id}, name={self.name})>"

# Konfiguracija enginea sa connection poolingom
engine = create_engine(
    DATABASE_URL,
    pool_size=5,  # Broj konekcija u poolu
    max_overflow=10,  # Maksimalan broj dodatnih konekcija
    pool_timeout=30,  # Timeout za dobijanje konekcije
    pool_recycle=1800,  # Recikliraj konekcije nakon 30 minuta
    echo=False  # Postavi na True za SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db() -> None:
    """Inicijalizuj bazu i kreiraj tabele"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def get_db() -> Generator:
    """Dependency za dobijanje DB sesije"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error occurred: {e}")
        raise
    finally:
        db.close()