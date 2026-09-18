"""Seed a minimal, repeatable demo dataset for the chatbot.

Run after scripts/init_database.py:
    python scripts/seed_demo.py

Demo credentials:
    admin1 / demo123
    demo_user / demo123
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "langraph pipeline"))

import auth
import db_service
from models import Product, get_session


DEMO_USERS = (("admin1", "demo123"), ("demo_user", "demo123"))
DEMO_PRODUCTS = [
    {
        "product_id": 900001,
        "name": "Vong tay da thach anh trang demo",
        "category": "Vong tay phong thuy",
        "material": ["Da thach anh"],
        "compatible_elements": ["Kim", "Thuy"],
        "colors": ["trang"],
        "product_size": ["6 li", "8 li"],
        "price_range": "199000-299000",
        "brand": "Van An Group",
        "origin": "Viet Nam",
        "warranty": "Bao hanh 6 thang",
        "in_stock": True,
        "quantity_min": 10,
        "quantity_max": 10,
        "product_description": "San pham demo dung de kiem thu chatbot phong thuy.",
        "image": None,
    },
    {
        "product_id": 900002,
        "name": "Vong tay da mat ho vang nau demo",
        "category": "Vong tay phong thuy",
        "material": ["Da mat ho"],
        "compatible_elements": ["Tho", "Kim"],
        "colors": ["vang", "nau"],
        "product_size": ["8 li", "10 li"],
        "price_range": "249000-349000",
        "brand": "Van An Group",
        "origin": "Viet Nam",
        "warranty": "Bao hanh 6 thang",
        "in_stock": True,
        "quantity_min": 8,
        "quantity_max": 8,
        "product_description": "San pham demo dung de kiem thu goi y theo menh Tho va Kim.",
        "image": None,
    },
]


def seed_users() -> None:
    engine = auth.get_engine()
    for username, password in DEMO_USERS:
        with engine.begin() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM users WHERE lower(username) = lower(:username)"),
                {"username": username},
            ).first()
        if exists:
            print(f"  User {username} already exists.")
            continue
        auth.register_user(username, password)
        print(f"  Created user {username}.")


def seed_products() -> None:
    session = get_session()
    try:
        for data in DEMO_PRODUCTS:
            product = session.query(Product).filter_by(product_id=data["product_id"]).one_or_none()
            if product is None:
                session.add(Product(**data))
                print(f"  Created product {data['product_id']}.")
                continue
            for field, value in data.items():
                setattr(product, field, value)
            print(f"  Updated product {data['product_id']}.")
        session.commit()
    finally:
        session.close()


def main() -> None:
    db_service.init_db()
    print("Seeding demo users...")
    seed_users()
    print("Seeding demo products...")
    seed_products()
    print("Demo data is ready. Product images are intentionally omitted, so no SigLIP reindex is needed.")


if __name__ == "__main__":
    main()