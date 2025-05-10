"""
Script di test per il rilevamento automatico dei keypoints del campo da padel.
Questo script utilizza le funzioni del modulo court_detection per testare
il rilevamento dei keypoints su un singolo frame.
"""
import cv2
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
import time
import torch
from pathlib import Path
import sys
import os

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.court_detection import auto_detect_court_keypoints, detect_court_keypoints_cv
from utils.keypoints_utils import draw_keypoints
from trackers import KeypointsTracker, Keypoint, Keypoints
from config import *

def test_cv_detection(image_path=None):
    """
    Testa il rilevamento dei keypoints usando solo tecniche di computer vision
    """
    if image_path is None:
        # Usa il video specificato in config.py
        print(f"Caricamento primo frame da {INPUT_VIDEO_PATH}")
        first_frame_generator = sv.get_video_frames_generator(
            INPUT_VIDEO_PATH,
            start=0,
            stride=1,
            end=1,
        )
        img = next(first_frame_generator)
    else:
        print(f"Caricamento immagine da {image_path}")
        img = cv2.imread(image_path)
    
    if img is None:
        print("Errore nel caricamento dell'immagine.")
        return
    
    print("Dimensioni immagine:", img.shape)
    
    print("Rilevamento keypoints con metodi CV...")
    start_time = time.time()
    keypoints = detect_court_keypoints_cv(img, debug=True)
    end_time = time.time()
    
    print(f"Tempo di elaborazione: {end_time - start_time:.2f} secondi")
    print(f"Rilevati {len(keypoints)} keypoints:")
    for i, kp in enumerate(keypoints):
        print(f"k{i+1}: {kp}")
    
    if keypoints:
        print("Disegno keypoints rilevati...")
        keypoints_img = draw_keypoints(img.copy(), keypoints)
        cv2.imshow('CV-Detected Court Keypoints', keypoints_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Salva l'immagine con i keypoints
        output_path = "debug_cv_keypoints.jpg"
        cv2.imwrite(output_path, keypoints_img)
        print(f"Immagine dei keypoints salvata in {output_path}")
    else:
        print("Nessun keypoint rilevato.")

def test_yolo_detection(image_path=None):
    """
    Testa il rilevamento dei keypoints usando il modello YOLO
    """
    if image_path is None:
        # Usa il video specificato in config.py
        print(f"Caricamento primo frame da {INPUT_VIDEO_PATH}")
        first_frame_generator = sv.get_video_frames_generator(
            INPUT_VIDEO_PATH,
            start=0,
            stride=1,
            end=1,
        )
        img = next(first_frame_generator)
    else:
        print(f"Caricamento immagine da {image_path}")
        img = cv2.imread(image_path)
    
    if img is None:
        print("Errore nel caricamento dell'immagine.")
        return
    
    print("Dimensioni immagine:", img.shape)
    
    print("Inizializzazione KeypointsTracker con modello YOLO...")
    try:
        # Inizializza il tracker di keypoints
        tracker = KeypointsTracker(
            model_path=KEYPOINTS_TRACKER_MODEL,
            batch_size=1,
            model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
            fixed_keypoints_detection=None,
        )
        
        # Usa GPU se disponibile
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilizzo device: {device}")
        tracker.to(device)
        
        print("Predizione keypoints...")
        start_time = time.time()
        predictions = tracker.predict_sample([img])
        end_time = time.time()
        
        print(f"Tempo di elaborazione: {end_time - start_time:.2f} secondi")
        
        if predictions and len(predictions[0].keypoints) > 0:
            keypoints = [(kp.xy[0], kp.xy[1]) for kp in predictions[0].keypoints]
            print(f"Rilevati {len(keypoints)} keypoints:")
            for i, kp in enumerate(keypoints):
                print(f"k{i+1}: {kp}")
            
            print("Disegno keypoints rilevati...")
            keypoints_img = draw_keypoints(img.copy(), keypoints)
            cv2.imshow('YOLO-Detected Court Keypoints', keypoints_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Salva l'immagine con i keypoints
            output_path = "debug_yolo_keypoints.jpg"
            cv2.imwrite(output_path, keypoints_img)
            print(f"Immagine dei keypoints salvata in {output_path}")
        else:
            print("Nessun keypoint rilevato dal modello YOLO.")
    
    except Exception as e:
        print(f"Errore durante il rilevamento con YOLO: {str(e)}")

def test_combined_detection(image_path=None):
    """
    Testa il rilevamento dei keypoints combinando YOLO e metodi CV
    """
    if image_path is None:
        # Usa il video specificato in config.py
        print(f"Caricamento primo frame da {INPUT_VIDEO_PATH}")
        first_frame_generator = sv.get_video_frames_generator(
            INPUT_VIDEO_PATH,
            start=0,
            stride=1,
            end=1,
        )
        img = next(first_frame_generator)
    else:
        print(f"Caricamento immagine da {image_path}")
        img = cv2.imread(image_path)
    
    if img is None:
        print("Errore nel caricamento dell'immagine.")
        return
    
    print("Dimensioni immagine:", img.shape)
    
    print("Inizializzazione KeypointsTracker con modello YOLO...")
    try:
        # Inizializza il tracker di keypoints
        tracker = KeypointsTracker(
            model_path=KEYPOINTS_TRACKER_MODEL,
            batch_size=1,
            model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
            fixed_keypoints_detection=None,
        )
        
        # Usa GPU se disponibile
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilizzo device: {device}")
        tracker.to(device)
        
        print("Predizione keypoints con YOLO...")
        predictions = tracker.predict_sample([img])
        
        yolo_keypoints = []
        if predictions and len(predictions[0].keypoints) > 0:
            yolo_keypoints = [(kp.xy[0], kp.xy[1]) for kp in predictions[0].keypoints]
            print(f"YOLO ha rilevato {len(yolo_keypoints)} keypoints")
            
            # Disegna i keypoints YOLO
            yolo_img = draw_keypoints(img.copy(), yolo_keypoints)
            cv2.imshow('YOLO Keypoints', yolo_img)
            cv2.waitKey(500)
        else:
            print("YOLO non ha rilevato alcun keypoint.")
        
        print("Rilevamento keypoints con metodi CV...")
        cv_keypoints = detect_court_keypoints_cv(img, debug=True)
        print(f"CV ha rilevato {len(cv_keypoints)} keypoints")
        
        # Disegna i keypoints CV
        if cv_keypoints:
            cv_img = draw_keypoints(img.copy(), cv_keypoints)
            cv2.imshow('CV Keypoints', cv_img)
            cv2.waitKey(500)
        
        print("Combinazione dei risultati...")
        
        # Usa la funzione auto_detect_court_keypoints che internamente
        # combina i risultati di YOLO e CV
        final_keypoints = auto_detect_court_keypoints(
            img, 
            yolo_model=tracker if yolo_keypoints else None,
            debug=True
        )
        
        print(f"Rilevamento finale: {len(final_keypoints)} keypoints")
        
        if final_keypoints:
            # Disegna i keypoints finali
            final_img = draw_keypoints(img.copy(), final_keypoints)
            cv2.imshow('Final Keypoints', final_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Salva l'immagine con i keypoints
            output_path = "debug_combined_keypoints.jpg"
            cv2.imwrite(output_path, final_img)
            print(f"Immagine dei keypoints salvata in {output_path}")
    
    except Exception as e:
        print(f"Errore durante il rilevamento combinato: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test di rilevamento keypoints del campo da padel')
    parser.add_argument('--method', choices=['cv', 'yolo', 'combined'], default='combined',
                        help='Metodo di rilevamento da testare (cv, yolo, o combined)')
    parser.add_argument('--image', type=str, default=None, 
                        help='Path all\'immagine da analizzare (opzionale, altrimenti usa il primo frame dal video in config.py)')
    
    args = parser.parse_args()
    
    if args.method == 'cv':
        test_cv_detection(args.image)
    elif args.method == 'yolo':
        test_yolo_detection(args.image)
    elif args.method == 'combined':
        test_combined_detection(args.image)
