""" General configurations for main.py """

# Input video path
INPUT_VIDEO_PATH = "/Users/claudio/Documents/GitHub/PadelStats/data/input/panasonic_final copia.mp4"

# Inference video path
OUTPUT_VIDEO_PATH = "results.mp4"

# True to collect 2d projection data
COLLECT_DATA = True
# Collected data path
COLLECT_DATA_PATH = "data.csv"

# Maximum number of frames to be analysed
MAX_FRAMES = None

# Fixed court keypoints
FIXED_COURT_KEYPOINTS_LOAD_PATH = None  # "/Users/claudio/Documents/GitHub/PadelStats/src/.cache/fixed_keypoints_detection.json"
FIXED_COURT_KEYPOINTS_SAVE_PATH = "/Users/claudio/Documents/GitHub/PadelStats/src/.cache/fixed_keypoints_detection.json"
# True to auto-detect court keypoints, False to select them manually
AUTO_DETECT_COURT_KEYPOINTS = True
# Metodo di rilevamento della corte: "traditional", "advanced", o "combined"
COURT_DETECTION_METHOD = "combined"
# True to verify and edit auto-detected keypoints
VERIFY_AUTO_DETECTED_KEYPOINTS = True

# Players tracker
PLAYERS_TRACKER_MODEL = "/Users/claudio/Documents/GitHub/PadelStats/src/.weights/players_detection/yolov8m.pt"
PLAYERS_TRACKER_BATCH_SIZE = 8
PLAYERS_TRACKER_ANNOTATOR = "rectangle_bounding_box"
PLAYERS_TRACKER_LOAD_PATH = None# "/Users/claudio/Documents/GitHub/PadelStats/src/.cache/players_detections.json"
PLAYERS_TRACKER_SAVE_PATH = "/Users/claudio/Documents/GitHub/PadelStats/src/.cache/players_detections.json"

# Players keypoints tracker
PLAYERS_KEYPOINTS_TRACKER_MODEL = "/Users/claudio/Documents/GitHub/PadelStats/src/.weights/players_keypoints_detection/best.pt"
PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE = 1280
PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE = 8
PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH = None#"/Users/claudio/Documents/GitHub/PadelStats/src/.cache/players_keypoints_detections.json"
PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH = "/Users/claudio/Documents/GitHub/PadelStats/src/.cache/players_keypoints_detections.json"

# Ball tracker
BALL_TRACKER_MODEL = "/Users/claudio/Documents/GitHub/PadelStats/src/.weights/ball_detection/TrackNet_best.pt"
BALL_TRACKER_INPAINT_MODEL = "/Users/claudio/Documents/GitHub/PadelStats/src/.weights/ball_detection/InpaintNet_best.pt"
BALL_TRACKER_BATCH_SIZE = 8
BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM = 400
BALL_TRACKER_LOAD_PATH = None #"/Users/claudio/Documents/GitHub/PadelStats/src/.cache/ball_detections.json"
BALL_TRACKER_SAVE_PATH = "/Users/claudio/Documents/GitHub/PadelStats/src/.cache/ball_detections.json"

# Court keypoints tracker
KEYPOINTS_TRACKER_MODEL = "/Users/claudio/Documents/GitHub/PadelStats/src/.weights/court_keypoints_detection/best.pt"
KEYPOINTS_TRACKER_BATCH_SIZE = 8
KEYPOINTS_TRACKER_MODEL_TYPE = "yolo"
KEYPOINTS_TRACKER_LOAD_PATH = None
KEYPOINTS_TRACKER_SAVE_PATH = None # "/Users/claudio/Documents/GitHub/PadelStats/src/.cache/keypoints_detections.json"

