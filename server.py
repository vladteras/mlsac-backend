from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import random
import sqlite3
import datetime
import os

app = FastAPI()

# Database setup
DB_FILE = "backend/mlsac.db"

import uuid

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS checks
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  player_uuid TEXT,
                  timestamp DATETIME,
                  probability REAL,
                  avg_jerk REAL,
                  verdict TEXT)''')
    
    # Create servers table
    c.execute('''CREATE TABLE IF NOT EXISTS servers
                 (id TEXT PRIMARY KEY,
                  name TEXT,
                  api_key TEXT,
                  license_type TEXT,
                  expiration_date DATETIME,
                  status TEXT,
                  online_count INTEGER DEFAULT 0,
                  limit_count INTEGER DEFAULT 50,
                  requests_today INTEGER DEFAULT 0,
                  detections_total INTEGER DEFAULT 0)''')
                  
    # Check if we have any servers, if not create a default Trial Server
    c.execute("SELECT count(*) FROM servers")
    if c.fetchone()[0] == 0:
        trial_id = str(uuid.uuid4())
        trial_key = "mls_live_" + str(uuid.uuid4()).replace("-", "")[:24]
        # Expire in 3 days
        expire_date = datetime.datetime.now() + datetime.timedelta(days=3)
        c.execute("INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (trial_id, "Trial Server", trial_key, "Trial period", expire_date, "Active", 0, 50, 0, 0))
        conn.commit()
        
    conn.commit()
    conn.close()

init_db()

class Server(BaseModel):
    id: str
    name: str
    api_key: str
    license_type: str
    expiration_date: datetime.datetime
    status: str
    online_count: int
    limit_count: int
    requests_today: int
    detections_total: int

class CreateServerRequest(BaseModel):
    name: str

class HeartbeatRequest(BaseModel):
    online_count: int

@app.post("/api/servers/{server_id}/heartbeat")
async def heartbeat(server_id: str, request: HeartbeatRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE servers SET online_count = ? WHERE id = ?", (request.online_count, server_id))
    if c.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Server not found")
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/servers")
async def get_servers():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM servers")
    servers = [dict(row) for row in c.fetchall()]
    conn.close()
    return servers

@app.post("/api/servers")
async def create_server(request: CreateServerRequest):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    new_id = str(uuid.uuid4())
    new_key = "mls_live_" + str(uuid.uuid4()).replace("-", "")[:24]
    expire_date = datetime.datetime.now() + datetime.timedelta(days=30) # Default 30 days
    
    c.execute("INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (new_id, request.name, new_key, "Standard License", expire_date, "Active", 0, 100, 0, 0))
    conn.commit()
    conn.close()
    return {"success": True, "id": new_id}

@app.get("/api/servers/{server_id}")
async def get_server(server_id: str):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM servers WHERE id = ?", (server_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Server not found")
        
    return dict(row)

@app.post("/api/servers/{server_id}/reset_key")
async def reset_key(server_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    new_key = "mls_live_" + str(uuid.uuid4()).replace("-", "")[:24]
    c.execute("UPDATE servers SET api_key = ? WHERE id = ?", (new_key, server_id))
    
    if c.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Server not found")
        
    conn.commit()
    conn.close()
    return {"api_key": new_key}

@app.get("/api/server/verify")
async def verify_server(key: str):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM servers WHERE api_key = ?", (key,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    return dict(row)

@app.post("/api/servers/{server_id}/renew")
async def renew_server(server_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Add 30 days
    c.execute("SELECT expiration_date FROM servers WHERE id = ?", (server_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Server not found")
        
    current_expire = datetime.datetime.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
    # If already expired, start from now. If not, add to current.
    if current_expire < datetime.datetime.now():
        new_expire = datetime.datetime.now() + datetime.timedelta(days=30)
    else:
        new_expire = current_expire + datetime.timedelta(days=30)
        
    c.execute("UPDATE servers SET expiration_date = ?, status = 'Active' WHERE id = ?", (new_expire, server_id))
    conn.commit()
    conn.close()
    return {"success": True, "expiration_date": new_expire}


class TickData(BaseModel):
    deltaYaw: float
    deltaPitch: float
    accelYaw: float
    accelPitch: float
    jerkYaw: float
    jerkPitch: float
    gcdErrorYaw: float
    gcdErrorPitch: float

class PredictRequest(BaseModel):
    playerUuid: str
    ticks: List[TickData]

@app.post("/predict")
async def predict(request: PredictRequest):
    # Simple heuristic logic (mock AI)
    avg_jerk = sum(abs(t.jerkYaw) + abs(t.jerkPitch) for t in request.ticks) / len(request.ticks) if request.ticks else 0
    
    probability = 0.0
    if avg_jerk > 0.5:
        probability = min(0.99, avg_jerk / 2.0)
    else:
        probability = random.uniform(0.0, 0.3)
    
    verdict = "LEGIT"
    if probability > 0.8:
        verdict = "CHEAT"
    elif probability > 0.5:
        verdict = "SUSPICIOUS"

    # Save to DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO checks (player_uuid, timestamp, probability, avg_jerk, verdict) VALUES (?, ?, ?, ?, ?)",
              (request.playerUuid, datetime.datetime.now(), probability, avg_jerk, verdict))
    conn.commit()
    conn.close()
    
    print(f"[{datetime.datetime.now()}] {request.playerUuid} -> Prob: {probability:.4f} ({verdict})")
    
    return {"probability": probability}

@app.get("/api/stats")
async def get_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Total checks
    c.execute("SELECT COUNT(*) FROM checks")
    total_checks = c.fetchone()[0]
    
    # High probability count (Cheaters)
    c.execute("SELECT COUNT(*) FROM checks WHERE probability > 0.8")
    flagged_count = c.fetchone()[0]
    
    # Recent checks (last 50)
    c.execute("SELECT player_uuid, timestamp, probability, verdict FROM checks ORDER BY id DESC LIMIT 50")
    recent_checks = [{"player": row[0], "time": row[1], "prob": row[2], "verdict": row[3]} for row in c.fetchall()]
    
    conn.close()
    
    return {
        "total_checks": total_checks,
        "flagged_count": flagged_count,
        "recent_checks": recent_checks
    }

# Serve static files
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('backend/static/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
