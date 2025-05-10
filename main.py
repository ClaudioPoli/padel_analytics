import timeit
import json
import cv2
import numpy as np
import supervision as sv
import torch

from trackers import (
    PlayerTracker, 
    BallTracker, 
    KeypointsTracker, 
    Keypoint,
    Keypoints,
    PlayerKeypointsTracker,
    TrackingRunner,
)
from utils.court_detection import auto_detect_court_keypoints, detect_court_keypoints_cv, improve_court_detection, orchestrate_court_detection
from utils.enhanced_court_detection import detect_court_keypoints_enhanced
from utils.robust_court_detection import robust_court_detection
from utils.keypoints_utils import draw_keypoints, manual_keypoints_selection
from config import *


SELECTED_KEYPOINTS = []

"""
PADEL COURT KEYPOINTS 

Automated detection or manual selection of the 12 keypoints:

        k11--------------------k12
        |                       |
        k8-----------k9--------k10
        |            |          |
        |            |          |
        |            |          |
        k6----------------------k7
        |            |          |
        |            |          |
        |            |          |
        k3-----------k4---------k5
        |                       |
        k1----------------------k2
        
"""

def click_event(event, x, y, flags, params): 
    """Legacy manual selection function"""
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        SELECTED_KEYPOINTS.append((x, y))
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('frame', img) 


if __name__ == "__main__":
    
    t1 = timeit.default_timer()

    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    fps, w, h, total_frames = (
        video_info.fps, 
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )

    first_frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=1,
    )

    img = next(first_frame_generator)

    # Try to load keypoints from file if specified
    if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None:
        print(f"Loading court keypoints from {FIXED_COURT_KEYPOINTS_LOAD_PATH}")
        with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:
            SELECTED_KEYPOINTS = json.load(f)
    # Auto-detect keypoints if enabled
    elif AUTO_DETECT_COURT_KEYPOINTS:
        print("Auto-detecting court keypoints...")
        try:
            # Utilizziamo il sistema robusto di rilevamento che gestisce meglio gli elementi di disturbo nell'immagine
            print("Rilevamento keypoints della corte usando il sistema robusto...")
            detected_keypoints = robust_court_detection(img, debug=True)
            
            # Se abbiamo keypoints ma non tutti i 12, e abbiamo YOLO disponibile, proviamo anche quello
            if AUTO_DETECT_COURT_KEYPOINTS and len(detected_keypoints) < 12 and KEYPOINTS_TRACKER_MODEL:
                print("Tentativo di rilevamento con YOLO...")
                # Initialize keypoints detector without fixed keypoints
                auto_keypoints_tracker = KeypointsTracker(
                    model_path=KEYPOINTS_TRACKER_MODEL,
                    batch_size=1,
                    model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
                    fixed_keypoints_detection=None,
                )
                
                # Put model on GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                auto_keypoints_tracker.to(device)
                
                # Process the first frame
                predictions = auto_keypoints_tracker.predict_sample([img])
                
                if predictions and len(predictions[0].keypoints) > 0:
                    # Extract detected keypoints
                    yolo_keypoints = [(kp.xy[0], kp.xy[1]) for kp in predictions[0].keypoints]
                    print(f"YOLO ha rilevato {len(yolo_keypoints)} keypoints")
                    
                    # Combina i keypoints CV e YOLO
                    from utils.court_detection import combine_keypoints
                    combined_keypoints = combine_keypoints(detected_keypoints, yolo_keypoints, img.shape)
                    if len(combined_keypoints) > len(detected_keypoints):
                        detected_keypoints = combined_keypoints
                        print(f"Combinazione riuscita: ora abbiamo {len(detected_keypoints)} keypoints")
                
                # Verifica e modifica manuale se richiesto
                if len(detected_keypoints) == 12:
                    print("Rilevati tutti i 12 keypoints richiesti.")
                    if VERIFY_AUTO_DETECTED_KEYPOINTS:
                        print("Per favore, verifica e modifica i keypoints rilevati se necessario.")
                        print("Premi SPAZIO quando hai finito.")
                        SELECTED_KEYPOINTS = manual_keypoints_selection(img, detected_keypoints)
                    else:
                        SELECTED_KEYPOINTS = detected_keypoints
                    
                    # Mostra i keypoints per ispezione
                    keypoints_img = draw_keypoints(img.copy(), SELECTED_KEYPOINTS)
                    cv2.imshow('Detected Court Keypoints', keypoints_img)
                    cv2.waitKey(1000)  # Mostra per 1 secondo
                    cv2.destroyAllWindows()
                else:
                    print(f"Rilevamento automatico non completo: {len(detected_keypoints)}/12 keypoints. Utilizzo della selezione manuale.")
                    SELECTED_KEYPOINTS = manual_keypoints_selection(img, detected_keypoints if len(detected_keypoints) > 0 else None)
                print("Rilevamento automatico fallito. Utilizzo della selezione manuale.")
                SELECTED_KEYPOINTS = manual_keypoints_selection(img)
        except Exception as e:
            print(f"Errore durante il rilevamento automatico: {str(e)}")
            print("Utilizzo della selezione manuale.")
            SELECTED_KEYPOINTS = manual_keypoints_selection(img)
    # Manual selection (fallback)
    else:
        print("Please select the 12 keypoints of the padel court.")
        print("Press SPACE when done.")
        SELECTED_KEYPOINTS = manual_keypoints_selection(img)

    # Save keypoints if path is specified
    if FIXED_COURT_KEYPOINTS_SAVE_PATH is not None:
        print(f"Saving court keypoints to {FIXED_COURT_KEYPOINTS_SAVE_PATH}")
        with open(FIXED_COURT_KEYPOINTS_SAVE_PATH, "w") as f:
            json.dump(SELECTED_KEYPOINTS, f)

    # Create keypoints object for tracker
    fixed_keypoints_detection = Keypoints(
        [
            Keypoint(
                id=i,
                xy=tuple(float(x) for x in v)
            )
            for i, v in enumerate(SELECTED_KEYPOINTS)
        ]
    )

    # Create polygon for tracking zone using keypoints
    # Utilizziamo il nuovo metodo per ottenere il poligono del campo
    try:
        from utils.update_court_detection import get_court_polygon
        polygon_vertices = get_court_polygon(SELECTED_KEYPOINTS)
    except (ImportError, Exception):
        # Fallback al metodo originale se quello nuovo non è disponibile
        keypoints_array = np.array(SELECTED_KEYPOINTS)
        polygon_vertices = np.concatenate(
            (
                np.expand_dims(keypoints_array[0], axis=0),  # k1 (in basso a sinistra)
                np.expand_dims(keypoints_array[1], axis=0),  # k2 (in basso a destra)
                np.expand_dims(keypoints_array[-1], axis=0), # k12 (in alto a destra)
                np.expand_dims(keypoints_array[-2], axis=0), # k11 (in alto a sinistra)
            ),
            axis=0
        )

    # Polygon to filter person detections inside padel court
    polygon_zone = sv.PolygonZone(
        polygon=polygon_vertices,
        #resolution_wh=video_info.resolution_wh,
    )


    # FILTER FRAMES OF INTEREST (TODO)


    # Instantiate trackers
    players_tracker = PlayerTracker(
        PLAYERS_TRACKER_MODEL,
        polygon_zone,
        batch_size=PLAYERS_TRACKER_BATCH_SIZE,
        annotator=PLAYERS_TRACKER_ANNOTATOR,
        show_confidence=True,
        load_path=PLAYERS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_TRACKER_SAVE_PATH,
    )

    player_keypoints_tracker = PlayerKeypointsTracker(
        PLAYERS_KEYPOINTS_TRACKER_MODEL,
        train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
        batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
        load_path=PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
    )
  
    ball_tracker = BallTracker(
        BALL_TRACKER_MODEL,
        BALL_TRACKER_INPAINT_MODEL,
        batch_size=BALL_TRACKER_BATCH_SIZE,
        median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
        median=None,
        load_path=BALL_TRACKER_LOAD_PATH,
        save_path=BALL_TRACKER_SAVE_PATH,
    )

    keypoints_tracker = KeypointsTracker(
        model_path=KEYPOINTS_TRACKER_MODEL,
        batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
        model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
        fixed_keypoints_detection=fixed_keypoints_detection,
        load_path=KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=KEYPOINTS_TRACKER_SAVE_PATH,
    )

    runner = TrackingRunner(
        trackers=[
            players_tracker, 
            player_keypoints_tracker, 
            ball_tracker,
            keypoints_tracker,    
        ],
        video_path=INPUT_VIDEO_PATH,
        inference_path=OUTPUT_VIDEO_PATH,
        start=0,
        end=MAX_FRAMES,
        collect_data=COLLECT_DATA,
    )

    runner.run()

    if COLLECT_DATA:
        data = runner.data_analytics.into_dataframe(runner.video_info.fps)
        data.to_csv(COLLECT_DATA_PATH)

    t2 = timeit.default_timer()

    print("Duration (min): ", (t2 - t1) / 60)
