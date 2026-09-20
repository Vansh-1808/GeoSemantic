import httpx

query = 'New construction near roads after 2023'
resp = httpx.post('http://localhost:8000/api/change/analyze', json={'query': query, 'limit': 10}, timeout=60.0)
print('Status:', resp.status_code)
data = resp.json()
print('Found:', data.get('candidates_found'))
for i, ev in enumerate(data.get('events', [])[:10]):
    ev_data = ev.get('evidence') or {}
    q_score = ev_data.get('query_match_score')
    b_date = (ev.get('before_date') or '')[:10]
    a_date = (ev.get('after_date') or '')[:10]
    expl = (ev_data.get('explanation') or '')[:80]
    print(f"#{i+1}: type={ev.get('change_type')} | conf={ev.get('final_confidence')} | q_score={q_score} | b_date={b_date} | a_date={a_date} | expl={expl}")
