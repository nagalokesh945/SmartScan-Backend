import base64
import json
import cv2
import numpy as np
import sqlite3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from insightface.app import FaceAnalysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting InsightFace Neural Network...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.40) 
print("AI Optics Online.")

DB_FILE = "biometrics.db"
KNOWN_FACES_CACHE = {}

def refresh_known_faces_cache():
    global KNOWN_FACES_CACHE
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT roll_number, name, embeddings FROM users")
    rows = cursor.fetchall()
    conn.close()

    updated_cache = {}
    for roll, name, emb_json in rows:
        updated_cache[roll] = {
            "name": name,
            "vector": np.array(json.loads(emb_json))
        }
    KNOWN_FACES_CACHE = updated_cache

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            roll_number TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embeddings TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    refresh_known_faces_cache()

init_db()

class RegistrationPayload(BaseModel):
    roll_number: str
    name: str
    images: List[str]
    force_update: bool = False
    old_roll_number: Optional[str] = None

def decode_b64(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.post("/register")
async def register_face(payload: RegistrationPayload):
    try:
        embeddings = []
        for img_b64 in payload.images:
            img = decode_b64(img_b64)
            faces = face_app.get(img)
            if faces:
                embeddings.append(faces[0].normed_embedding)
        
        if not embeddings:
            return {"status": "error", "message": "AI failed to map facial geometry in captures."}
        
        master_vector = np.mean(embeddings, axis=0)
        master_vector = master_vector / np.linalg.norm(master_vector)
        vector_json = json.dumps(master_vector.tolist())
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        if not payload.force_update:
            cursor.execute("SELECT roll_number, name, embeddings FROM users")
            rows = cursor.fetchall()

            best_match_roll = None
            best_match_name = None
            highest_sim = 0.0

            for db_roll, db_name, emb_json in rows:
                db_vector = np.array(json.loads(emb_json))
                sim = float(np.dot(master_vector, db_vector))
                if sim > highest_sim:
                    highest_sim = sim
                    best_match_roll = db_roll
                    best_match_name = db_name

            if highest_sim > 0.40 and best_match_roll:
                conn.close()
                return {
                    "status": "duplicate",
                    "existing_roll": best_match_roll,
                    "existing_name": best_match_name,
                    "similarity": round(highest_sim * 100, 1),
                    "message": f"This face is already registered under {best_match_name} ({best_match_roll})."
                }

        if payload.force_update and payload.old_roll_number and payload.old_roll_number != payload.roll_number:
            cursor.execute("DELETE FROM users WHERE roll_number = ?", (payload.old_roll_number,))

        cursor.execute(
            "INSERT OR REPLACE INTO users (roll_number, name, embeddings) VALUES (?, ?, ?)",
            (payload.roll_number, payload.name, vector_json)
        )
        conn.commit()
        conn.close()

        refresh_known_faces_cache()
        return {"status": "success", "message": f"Biometrics encrypted for {payload.name} ({payload.roll_number})"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/users")
async def get_users():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT roll_number, name, embeddings FROM users")
        rows = cursor.fetchall()
        conn.close()
        return [{"roll_number": r[0], "name": r[1], "embeddings": r[2][:80] + "... [TRUNCATED]"} for r in rows]
    except Exception as e:
        return []

@app.delete("/users/{roll_number}")
async def delete_user(roll_number: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE roll_number = ?", (roll_number,))
        conn.commit()
        conn.close()
        refresh_known_faces_cache()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/cctv")
async def cctv_scanner(websocket: WebSocket):
    await websocket.accept()
    refresh_known_faces_cache()
    
    # NEW: Track who has been adapted this session to prevent SQLite locking lag
    session_adapted_faces = set()

    try:
        while True:
            data = await websocket.receive_text()
            img = decode_b64(data)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if texture_score < 10.0:
                await websocket.send_json({"scanned": [], "unknown": True})
                continue 
                
            faces = face_app.get(img)
            recognized = []
            unknown = False
            
            if len(faces) == 0:
                await websocket.send_json({"scanned": [], "unknown": False})
                continue
            
            for face in faces:
                live_vector = face.normed_embedding
                best_match_roll = None
                highest_sim = 0.0
                
                for roll, db_data in KNOWN_FACES_CACHE.items():
                    sim = float(np.dot(live_vector, db_data["vector"]))
                    if sim > highest_sim:
                        highest_sim = sim
                        best_match_roll = roll
                
                if highest_sim > 0.38 and best_match_roll:
                    recognized.append(best_match_roll)

                    # --- CONTINUOUS ADAPTIVE LEARNING ---
                    # If similarity is very high, average their DB vector with their current face 
                    # so the system learns their aging/hairstyles over time.
                    if highest_sim > 0.85 and best_match_roll not in session_adapted_faces:
                        old_vec = KNOWN_FACES_CACHE[best_match_roll]["vector"]
                        
                        # Apply moving average (90% old face, 10% new face)
                        new_vec = (old_vec * 0.9) + (live_vector * 0.1)
                        new_vec = new_vec / np.linalg.norm(new_vec)
                        
                        # Update RAM cache instantly
                        KNOWN_FACES_CACHE[best_match_roll]["vector"] = new_vec
                        
                        # Async push to SQLite
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute("UPDATE users SET embeddings = ? WHERE roll_number = ?", 
                                       (json.dumps(new_vec.tolist()), best_match_roll))
                        conn.commit()
                        conn.close()
                        
                        # Lock adaptation for this session
                        session_adapted_faces.add(best_match_roll)
                        print(f"AI Adapted: Progressively updated face profile for {best_match_roll} (Sim: {highest_sim:.2f})")

                else:
                    unknown = True
            
            await websocket.send_json({"scanned": recognized, "unknown": unknown})
            
    except WebSocketDisconnect:
        print("CCTV Feed Disconnected.")