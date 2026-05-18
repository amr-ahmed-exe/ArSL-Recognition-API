from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import json
from typing import List
import os
import asyncio
import logging
from collections import Counter, deque
from spellchecker import SpellChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArSL")

spell = SpellChecker(language='ar')

def get_suggestions(word: str, max_count: int = 4) -> list[str]:
    word = word.strip()
    if not word: return []
    try:
        candidates = spell.candidates(word)
        if candidates is None:
            return []
        
        candidates_list = list(candidates)
        filtered = [s for s in candidates_list if s.startswith(word)]
        if not filtered:
            filtered = candidates_list
            
        return filtered[:max_count]
    except Exception:
        return []

app = FastAPI(title="ArSL API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model & encoder on startup ──────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",   "arsl_rf_model.joblib")
ENCODER_PATH = os.getenv("ENCODER_PATH", "arsl_label_encoder.joblib")

model = joblib.load(MODEL_PATH)
le    = joblib.load(ENCODER_PATH)
CLASS_LIST = list(le.classes_)
N_FEATURES = model.n_features_in_
print(f"✅ Model loaded | Classes: {len(le.classes_)} | Features: {N_FEATURES}")

# Detect whether the model was trained WITH geometric features (v2, ~115 features)
# or WITHOUT (v1, 89 features). This lets the API work with both old and new models.
USE_GEOMETRIC_FEATURES = N_FEATURES > 89
if USE_GEOMETRIC_FEATURES:
    print("✅ Model includes geometric features — disambiguation post-processing DISABLED")
else:
    print("⚠️  Model does NOT include geometric features — using legacy 89-feature pipeline")


# ── Arabic letter mapping (English label → Arabic character) ─────────────────
LABEL_TO_ARABIC = {
    "ain":   "ع",
    "al":    "ال",
    "aleff": "ا",
    "bb":    "ب",
    "dal":   "د",
    "dha":   "ظ",
    "dhad":  "ض",
    "fa":    "ف",
    "gaaf":  "ق",
    "ghain": "غ",
    "ha":    "ه",
    "haa":   "ح",
    "jeem":  "ج",
    "kaaf":  "ك",
    "khaa":  "خ",
    "la":    "لا",
    "laam":  "ل",
    "meem":  "م",
    "nun":   "ن",
    "ra":    "ر",
    "saad":  "ص",
    "seen":  "س",
    "sheen": "ش",
    "ta":    "ط",
    "taa":   "ت",
    "thaa":  "ث",
    "thal":  "ذ",
    "toot":  "ة",
    "waw":   "و",
    "ya":    "ي",
    "yaa":   "ى",
    "zay":   "ز",
}

def to_arabic(label: str) -> str:
    """Convert an English transliterated label to its Arabic character."""
    return LABEL_TO_ARABIC.get(label, label)


# ── Feature engineering (must match training exactly) ────────────────────────
def normalize_landmarks(raw: List[float]) -> np.ndarray:
    """
    raw: flat list of 63 floats  [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
    Returns normalized (63,) array with rotation normalization.
    """
    pts = np.array(raw, dtype=np.float32).reshape(21, 3)
    wrist = pts[0].copy()
    pts   = pts - wrist                                  # translate
    scale = np.linalg.norm(pts[9])                       # wrist→middle_mcp
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale

    # ── 2D Rotation Normalization ─────────────────────────────────────
    # Align wrist→middle_mcp to always point in negative Y direction
    # (fingers pointing up in image coordinates)
    # This makes features invariant to hand rotation in the camera plane
    ref = pts[9, :2]                                     # x, y of middle MCP
    ref_len = np.linalg.norm(ref)
    if ref_len > 1e-6:
        cos_a = -ref[1] / ref_len
        sin_a = -ref[0] / ref_len
        rot = np.array([[cos_a, -sin_a],
                         [sin_a,  cos_a]], dtype=np.float32)
        pts[:, :2] = (rot @ pts[:, :2].T).T

    return pts.flatten()                                 # (63,)


def add_distance_features(pts: np.ndarray) -> np.ndarray:
    """
    pts: normalized (63,) → returns (89,) with 26 extra distance/curl features.
    """
    p = pts.reshape(21, 3)

    def dist(a, b):
        return float(np.linalg.norm(p[a] - p[b]))

    feats = list(pts)

    fingertips = [4, 8, 12, 16, 20]

    # Fingertip → wrist (5)
    for tip in fingertips:
        feats.append(dist(tip, 0))

    # Fingertip → fingertip all pairs (10)
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            feats.append(dist(fingertips[i], fingertips[j]))

    # Fingertip → its MCP (5)
    for tip, mcp in [(4,1),(8,5),(12,9),(16,13),(20,17)]:
        feats.append(dist(tip, mcp))

    # Thumb → index pip/dip (2)
    feats.append(dist(4, 7))
    feats.append(dist(4, 6))

    # Curl ratios (4)
    finger_joints = [
        (8,  6,  5),   # index:  tip, pip, mcp
        (12, 10, 9),   # middle
        (16, 14, 13),  # ring
        (20, 18, 17),  # pinky
    ]
    for tip, pip, mcp in finger_joints:
        tip_mcp = dist(tip, mcp)
        pip_mcp = dist(pip, mcp)
        feats.append(tip_mcp / (pip_mcp + 1e-6))

    return np.array(feats, dtype=np.float32)   # (89,)


def add_geometric_features(pts: np.ndarray) -> np.ndarray:
    """
    pts: normalized (63,) → adds geometric hand-shape features.
    Must match the training pipeline exactly.
    
    These features enable the model to distinguish visually similar signs
    (e.g., ain/taa, jeem/haa/khaa, dal/thal, seen/sheen, etc.)
    without needing any post-processing heuristics.
    """
    p = pts.reshape(21, 3)

    def dist(a, b):
        return float(np.linalg.norm(p[a] - p[b]))

    def dist_from_origin(i):
        return float(np.linalg.norm(p[i]))

    feats = []

    # ── 1. Finger Extension (6 features) ─────────────────────────────
    def is_extended(tip, pip):
        return 1.0 if dist_from_origin(tip) > dist_from_origin(pip) else 0.0

    thumb_ext = 1.0 if dist_from_origin(4) > dist_from_origin(3) * 1.05 else 0.0
    index_ext = is_extended(8, 6)
    middle_ext = is_extended(12, 10)
    ring_ext = is_extended(16, 14)
    pinky_ext = is_extended(20, 18)
    ext_count = index_ext + middle_ext + ring_ext + pinky_ext

    feats.extend([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext, ext_count])

    # ── 2. Thumb curl ratio (1 feature) ──────────────────────────────
    thumb_tip_mcp = dist(4, 2)
    thumb_ip_mcp = dist(3, 2)
    feats.append(thumb_tip_mcp / (thumb_ip_mcp + 1e-6))

    # ── 3. Adjacent finger spreads (4 features) ──────────────────────
    feats.append(dist(4, 8))    # thumb-index
    feats.append(dist(8, 12))   # index-middle
    feats.append(dist(12, 16))  # middle-ring
    feats.append(dist(16, 20))  # ring-pinky

    # ── 4. Thumb position features (4 features) ──────────────────────
    feats.append(float(p[4][2] - p[5][2]))  # thumb depth (z vs index MCP)
    feats.append(float(p[4][1] - p[5][1]))  # thumb height (y vs index MCP)

    # Thumb-to-gap (distance from thumb tip to midpoint of index_pip & middle_pip, 2D)
    gap_mid = (p[6][:2] + p[10][:2]) / 2
    feats.append(float(np.linalg.norm(p[4][:2] - gap_mid)))

    # Thumb angle
    thumb_vec = p[4] - p[3]
    feats.append(float(np.arctan2(thumb_vec[1], thumb_vec[0])))

    # ── 5. Inter-finger angles (2 features) ──────────────────────────
    def angle_at(a, vertex, b):
        v1 = p[a] - p[vertex]
        v2 = p[b] - p[vertex]
        cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))

    feats.append(angle_at(8, 5, 12))   # index-middle angle
    feats.append(angle_at(12, 9, 16))  # middle-ring angle

    # ── 6. Hand openness (1 feature) ─────────────────────────────────
    feats.append((dist(8, 12) + dist(12, 16) + dist(16, 20)) / 3)

    # ── 7. Thumb-to-fingertip distances (3 features) ─────────────────
    feats.append(dist(4, 8))
    feats.append(dist(4, 12))
    feats.append(dist(4, 16))

    return np.array(feats, dtype=np.float32)


def build_features(landmarks: List[float]) -> np.ndarray:
    """
    Full feature engineering pipeline: normalize → distance features → geometric features.
    Returns feature vector ready for model.predict().
    """
    norm = normalize_landmarks(landmarks)
    feats_89 = add_distance_features(norm)  # (89,)
    
    if USE_GEOMETRIC_FEATURES:
        geo_feats = add_geometric_features(norm)  # (~26,)
        full = np.concatenate([feats_89, geo_feats])
    else:
        full = feats_89
    
    return full.reshape(1, -1)


# ── Request / Response schemas ────────────────────────────────────────────────
class LandmarkRequest(BaseModel):
    landmarks: List[float] = Field(
        ...,
        min_length=63,
        max_length=63,
        description="Flat list of 63 floats: [x0,y0,z0 ... x20,y20,z20]"
    )

class PredictionResponse(BaseModel):
    letter:     str
    confidence: float
    top3:       List[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "classes": len(le.classes_),
        "features": N_FEATURES,
        "geometric_features": USE_GEOMETRIC_FEATURES,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: LandmarkRequest):
    try:
        feats = build_features(req.landmarks)

        pred_idx  = model.predict(feats)[0]
        probs     = model.predict_proba(feats)[0]

        raw_label  = str(le.inverse_transform([pred_idx])[0])
        letter     = to_arabic(raw_label)
        confidence = float(round(probs[pred_idx], 4))

        # Top-3
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"letter": to_arabic(str(le.inverse_transform([i])[0])), "confidence": float(round(probs[i], 4))}
            for i in top3_idx
        ]

        return PredictionResponse(letter=letter, confidence=confidence, top3=top3)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    # ── Detection tuning constants ────────────────────────────────────
    HISTORY_SIZE = 9                # sliding window for smoothing
    MIN_CONFIDENCE = 0.40           # ignore predictions below this
    HIGH_CONF_THRESHOLD = 0.85      # very confident → fewer frames needed
    CONSEC_HIGH = 3                 # frames needed when confidence >= 85%
    CONSEC_NORMAL = 5               # frames needed otherwise
    NO_HAND_FRAMES_FOR_SPACE = 8    # hand-absent frames to trigger space

    # ── State variables ───────────────────────────────────────────────
    history = deque(maxlen=HISTORY_SIZE)       # (raw_label, confidence) tuples
    current_word = ""
    current_sentence = ""
    last_confirmed_letter = None
    consecutive_count = 0
    no_hand_count = 0              # consecutive frames with no hand
    hand_was_present = False       # whether we ever saw a hand

    last_word_for_suggestions = ""
    cached_suggestions = []

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                if data == "CLEAR":
                    current_word = ""
                    current_sentence = ""
                    history.clear()
                    last_confirmed_letter = None
                    consecutive_count = 0
                    no_hand_count = 0
                    hand_was_present = False
                    continue
                elif data.startswith("COMMIT:"):
                    selected_word = data.split(":", 1)[1]
                    current_sentence += (" " + selected_word) if current_sentence else selected_word
                    current_word = ""
                    history.clear()
                    consecutive_count = 0
                    continue
                else:
                    continue

            if "action" in payload:
                action = payload["action"]
                if action == "CLEAR":
                    current_word = ""
                    current_sentence = ""
                    history.clear()
                    last_confirmed_letter = None
                    consecutive_count = 0
                    no_hand_count = 0
                    hand_was_present = False
                elif action.startswith("COMMIT:"):
                    selected_word = action.split(":", 1)[1]
                    current_sentence += (" " + selected_word) if current_sentence else selected_word
                    current_word = ""
                    history.clear()
                    consecutive_count = 0
                elif action == "BACKSPACE":
                    if current_word:
                        current_word = current_word[:-1]
                    history.clear()
                    consecutive_count = 0
                continue

            # ── Hand-absent detection (no landmarks → space) ──────────
            if "landmarks" not in payload or payload.get("no_hand", False):
                if hand_was_present:
                    no_hand_count += 1
                    if no_hand_count == NO_HAND_FRAMES_FOR_SPACE and current_word:
                        # Hand removed → commit current word as space
                        current_sentence += (" " + current_word) if current_sentence else current_word
                        current_word = ""
                        history.clear()
                        last_confirmed_letter = None
                        consecutive_count = 0

                        display_text = current_sentence
                        if current_word:
                            display_text += (" " + current_word) if display_text else current_word

                        await websocket.send_json({
                            "status": "no_hand_space",
                            "raw_prediction": None,
                            "confirmed_letter": " ",
                            "current_word": current_word,
                            "final_word": display_text,
                            "suggestions": [],
                            "letter": None,
                            "confidence": 0.0,
                            "top3": [],
                            "hand_detected": False
                        })
                    elif no_hand_count > NO_HAND_FRAMES_FOR_SPACE:
                        # Keep sending no-hand status without re-triggering space
                        display_text = current_sentence
                        if current_word:
                            display_text += (" " + current_word) if display_text else current_word
                        await websocket.send_json({
                            "status": "no_hand",
                            "raw_prediction": None,
                            "confirmed_letter": None,
                            "current_word": current_word,
                            "final_word": display_text,
                            "suggestions": [],
                            "letter": None,
                            "confidence": 0.0,
                            "top3": [],
                            "hand_detected": False
                        })
                continue

            landmarks = payload["landmarks"]

            # Check expected size
            if len(landmarks) != 63:
                await websocket.send_json({"error": "Expected 63 landmarks"})
                continue

            # Hand is present → reset no-hand counter
            hand_was_present = True
            no_hand_count = 0

            feats = build_features(landmarks)

            # Run sync in threadpool
            pred_idx = await loop.run_in_executor(None, lambda: model.predict(feats)[0])
            probs = await loop.run_in_executor(None, lambda: model.predict_proba(feats)[0])

            raw_label = str(le.inverse_transform([pred_idx])[0])
            confidence = float(probs[pred_idx])
            letter = to_arabic(raw_label)
            status = "success"
            smoothed_letter = None
            confirmed_letter = None

            # ── Confidence-weighted smoothing ─────────────────────────
            if confidence >= MIN_CONFIDENCE:
                history.append((raw_label, confidence))

                # Weighted voting: each vote is weighted by its confidence
                vote_weights = {}
                for lbl, conf in history:
                    vote_weights[lbl] = vote_weights.get(lbl, 0.0) + conf
                smoothed_letter = max(vote_weights, key=vote_weights.get)

                # Adaptive consecutive count
                if smoothed_letter == last_confirmed_letter:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                    last_confirmed_letter = smoothed_letter

                # Determine required frames based on confidence
                avg_conf = vote_weights[smoothed_letter] / max(
                    sum(1 for lbl, _ in history if lbl == smoothed_letter), 1
                )
                required = CONSEC_HIGH if avg_conf >= HIGH_CONF_THRESHOLD else CONSEC_NORMAL

                if consecutive_count >= required:
                    confirmed_letter = to_arabic(smoothed_letter)
                    if smoothed_letter in ["Space", "space"]:
                        current_sentence += (" " + current_word) if current_sentence else current_word
                        current_word = ""
                    elif smoothed_letter in ["del", "nothing"]:
                        pass
                    else:
                        current_word += confirmed_letter

                    history.clear()
                    consecutive_count = 0
            else:
                # Low confidence → don't add to history, acts as noise filter
                pass

            # Suggestion logic
            if current_word and len(current_word) >= 2:
                if current_word != last_word_for_suggestions:
                    cached_suggestions = get_suggestions(current_word)
                    last_word_for_suggestions = current_word
                suggestions = cached_suggestions
            else:
                suggestions = []
                last_word_for_suggestions = ""

            display_text = current_sentence
            if current_word:
                display_text += (" " + current_word) if display_text else current_word

            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = [
                {"letter": to_arabic(str(le.inverse_transform([i])[0])), "confidence": float(round(probs[i], 4))}
                for i in top3_idx
            ]

            await websocket.send_json({
                "status": status,
                "raw_prediction": letter,
                "confirmed_letter": confirmed_letter,
                "current_word": current_word,
                "final_word": display_text,
                "suggestions": suggestions,
                "letter": letter,
                "confidence": float(round(confidence, 4)),
                "top3": top3,
                "hand_detected": True
            })

    except WebSocketDisconnect:
        print("Frontend WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
