from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "test_videos" / "hamburguer77.mp4"

# Output
SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = PROJECT_ROOT / "outputs" / "annotated_output.mp4"

# Classification
CONFIDENCE_THRESHOLD = 0.80

# Motion / event detection
MOTION_THRESHOLD = 35          # threshold do pixel diff
MIN_CONTOUR_AREA = 6000        # área mínima para considerar movimento
EVENT_END_DELAY_FRAMES = 8     # nº de frames sem movimento para fechar evento
MIN_EVENT_FRAMES = 3           # nº mínimo de frames válidas para aceitar evento

# ROI (ajustar conforme o vídeo)
ROI_X = 300
ROI_Y = 120
ROI_W = 740
ROI_H = 500

# Display
SHOW_VIDEO = True
FRAME_RESIZE = None  # ex: (960, 540) se quisere redimensionar