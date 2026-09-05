import asyncio
import sys
from pathlib import Path
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.db.database import get_session, AsyncSessionLocal
from sqlalchemy import text

async def verify_db():
    try:
        async with AsyncSessionLocal() as session:
            # Check Postgres
            res = await session.execute(text("SELECT version();"))
            pg_version = res.scalar()
            print(f"[OK] PostgreSQL connection works: {pg_version.split(',')[0]}")
            
            # Check PostGIS
            res = await session.execute(text("SELECT postgis_version();"))
            postgis_version = res.scalar()
            print(f"[OK] PostGIS connection works: {postgis_version}")
    except Exception as e:
        print(f"[FAIL] Database connection failed: {e}")

def verify_qdrant():
    try:
        # In our env.example we defined QDRANT_STORE_DIR
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        qdrant_dir = os.environ.get("QDRANT_STORE_DIR", "../data/qdrant_store")
        # Ensure it exists
        os.makedirs(qdrant_dir, exist_ok=True)
        
        client = QdrantClient(path=qdrant_dir)
        collections = client.get_collections()
        print(f"[OK] Qdrant local connection works. Collections found: {len(collections.collections)}")
    except Exception as e:
        print(f"[FAIL] Qdrant connection failed: {e}")

async def main():
    await verify_db()
    verify_qdrant()

if __name__ == "__main__":
    asyncio.run(main())
