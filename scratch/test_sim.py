import asyncio
from app.db.database import AsyncSessionLocal
from app.models.scene import Scene
from app.models.tile import Tile
from sqlalchemy import select

async def simulate():
    async with AsyncSessionLocal() as db:
        scenes = (await db.scalars(select(Scene))).all()
        print(f"Loaded {len(scenes)} scenes.")
        # Check Jewar scenes
        jewar_scenes = [s for s in scenes if "jewar" in s.filename.lower()]
        print("Jewar scenes count:", len(jewar_scenes))
        for js in jewar_scenes:
            print("  ", js.filename, js.acquisition_date)

if __name__ == "__main__":
    asyncio.run(simulate())
