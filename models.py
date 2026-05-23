"""
models.py – SQLAlchemy ORM models for PostgreSQL.

Tables:
  - products:    structured metadata for each product (Filter Search)
  - boilerplate: shared content (care instructions, warranty) by product type
"""

from sqlalchemy import (
    Column, Integer, String, Text, Boolean,
    ARRAY, create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import config


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    id                  = Column(Integer, primary_key=True, index=True)
    product_id          = Column(Integer, unique=True, nullable=False, index=True)
    name                = Column(String(500), nullable=False)
    category            = Column(String(100), nullable=False, index=True)
    material            = Column(ARRAY(String), default=[])
    compatible_elements = Column(ARRAY(String), default=[])
    colors              = Column(ARRAY(String), default=[])
    product_size        = Column(ARRAY(String), default=[])
    price_range         = Column(String(50), nullable=True)       # e.g., "100k-200k"
    brand               = Column(String(100), default="Vạn An Group")
    origin              = Column(String(100), default="Việt Nam")
    warranty            = Column(String(200), nullable=True)
    in_stock            = Column(Boolean, default=True)
    quantity_min        = Column(Integer, nullable=True)           # imported from so luong san pham.txt
    quantity_max        = Column(Integer, nullable=True)
    product_description = Column(Text, nullable=True)              # full product description
    image               = Column(JSONB, nullable=True)             # list of image URLs

    def __repr__(self):
        return f"<Product #{self.product_id}: {self.name[:50]}>"


class Boilerplate(Base):
    """
    Shared boilerplate content (care instructions, commitments)
    grouped by material/category. Lookup when chatbot needs it.
    """
    __tablename__ = "boilerplate"

    id            = Column(Integer, primary_key=True)
    material_type = Column(String(100), nullable=False, index=True)
    section_type  = Column(String(50), nullable=False)     # "bảo_quản" | "cam_kết"
    content       = Column(Text, nullable=False)

    def __repr__(self):
        return f"<Boilerplate {self.material_type}/{self.section_type}>"


# ═══════════════════════════════════════════════════════════════════
#  ENGINE & SESSION FACTORY
# ═══════════════════════════════════════════════════════════════════
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(config.PG_URL, echo=False, pool_pre_ping=True)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def create_tables():
    """Create all tables (idempotent)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("  ✓ PostgreSQL tables created / verified")


def drop_tables():
    """Drop all tables (for dev reset)."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("  ✓ PostgreSQL tables dropped")
