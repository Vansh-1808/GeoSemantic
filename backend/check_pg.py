import psycopg2
from app.core.config import settings
url = str(settings.database_sync_url).replace('+psycopg2','')
conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute("SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name LIKE '%postgis%';")
print("PostGIS Extensions:", cur.fetchall())
cur.close()
conn.close()
