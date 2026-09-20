import httpx

query = 'New construction near roads after 2023'
resp = httpx.post('http://localhost:8000/api/change/analyze', json={'query': query, 'limit': 30}, timeout=60.0)
data = resp.json()
print('Total returned:', data.get('candidates_found'))
for i, ev in enumerate(data.get('events', [])):
    ev_data = ev.get('evidence') or {}
    q_score = ev_data.get('query_match_score')
    b_date = (ev.get('before_date') or '')[:10]
    a_date = (ev.get('after_date') or '')[:10]
    expl = (ev_data.get('explanation') or '')[:60]
    b_thumb = ev_data.get('before_thumbnail_url') or ''
    print(f"#{i+1}: conf={ev.get('final_confidence')} | q_score={q_score} | dates={b_date}->{a_date} | thumb={b_thumb[:40]} | expl={expl}")
