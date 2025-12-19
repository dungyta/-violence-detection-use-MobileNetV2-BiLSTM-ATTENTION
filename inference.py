import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from model import ViolenceModel

# ================== CONFIG ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 16
THRESHOLD = 0.55
VIDEO_OUT = "output_violence_demo.mp4"
MODEL_PATH = "best_model.pth"

# ================== TRANSFORM ==================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# ================== EMA ==================
class EMA:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * x
        return self.value

# ================== VISUAL ==================
def draw_confidence(frame, conf):
    text = f"Violence: {conf:.2f}"
    color = (0, 0, 255) if conf > THRESHOLD else (0, 255, 0)

    cv2.rectangle(frame, (10, 10), (240, 55), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def draw_border(frame, conf):
    if conf > THRESHOLD:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 6)

# ================== LOAD MODEL ==================
model = ViolenceModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

ema = EMA(alpha=0.8)

# ================== INFERENCE ==================
def infer_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb)
        buffer.append(tensor)

        if len(buffer) > SEQ_LEN:
            buffer.pop(0)

        conf = 0.0

        if len(buffer) == SEQ_LEN:
            clip = torch.stack(buffer).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(clip)
                prob = F.softmax(logits, dim=1)[0, 1].item()
                conf = ema.update(prob)

        # ===== DRAW =====
        draw_confidence(frame, conf)
        draw_border(frame, conf)

        cv2.imshow("Violence Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ================== RUN ==================
if __name__ == "__main__":
    infer_video("/home/dun/project_violence/archive/data/fi1_xvid.avi")
