import sys, os
sys.path.insert(0, ".")
import asyncio
from app.db.database import AsyncSessionLocal
from app.services.change_detection import ChangeDetectionService
from app.schemas.change import ChangeAnalyzeRequest

async def test():
    async with AsyncSessionLocal() as db:
        service = ChangeDetectionService()
        req = ChangeAnalyzeRequest(
            query='New construction near roads after 2023"',
            limit=10
        )
        count, events = await service.analyze_change(req, db)
        print("Total events returned:", count)
        for i, ev in enumerate(events):
            q_score = ev.evidence.get("query_match_score") if ev.evidence else None
            expl = (ev.evidence.get("explanation") or "")[:80] if ev.evidence else ""
            b_d = str(ev.before_date)[:10] if ev.before_date else ""
            a_d = str(ev.after_date)[:10] if ev.after_date else ""
            print(f"#{i+1}: type={ev.change_type} | conf={ev.final_confidence:.3f} | q_score={q_score} | b_date={b_d} | a_date={a_d} | expl={expl}")

if __name__ == "__main__":
    asyncio.run(test())
