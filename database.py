# Database module with PostgreSQL support
# Replaces SQLite with proper production database

import os
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback to SQLite for local dev (with proper settings)
    DATABASE_URL = f"sqlite:///{os.getenv('DB_PATH', 'vividmedi.db')}"
    # For SQLite: disable WAL and use timeout
    engine = create_engine(
        DATABASE_URL,
        connect_args={"timeout": 30, "check_same_thread": False},
        pool_pre_ping=True,
        echo=False
    )
else:
    # PostgreSQL for production
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=40,
        pool_recycle=3600,
        pool_pre_ping=True,
        echo=False
    )

@contextmanager
def db_conn():
    """Context manager for database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def db_init():
    """Initialize database tables"""
    with db_conn() as conn:
        # Create extensions (PostgreSQL only)
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS uuid-ossp;"))
        except:
            pass
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                picture TEXT,
                plan TEXT NOT NULL DEFAULT 'free',
                email_verified INTEGER NOT NULL DEFAULT 1,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                specialty TEXT,
                expertise_level TEXT,
                created_at TEXT NOT NULL
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS usage (
                actor_type TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (actor_type, actor_id)
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                mode TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))
        
        try:
            conn.commit()
        except:
            conn.rollback()

def execute_query(query: str, params: dict = None):
    """Execute a query and return results"""
    with db_conn() as conn:
        result = conn.execute(text(query), params or {})
        return result.fetchall()

def execute_update(query: str, params: dict = None):
    """Execute an update/insert/delete query"""
    with db_conn() as conn:
        conn.execute(text(query), params or {})
        conn.commit()
