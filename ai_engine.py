import cv2
from insightface.app import FaceAnalysis

ai_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) 
ai_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)

def process_cctv_frame(frame):
    faces = ai_app.get(frame)
    return [{"bbox": f.bbox.astype(int).tolist(), "embedding": f.embedding} for f in faces]