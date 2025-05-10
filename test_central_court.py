#!/usr/bin/env python
"""
Script per creare un filtro avanzato per rilevare solo il campo centrale da padel, eliminando le zone laterali.
Questo script testa l'algoritmo migliorato su diverse immagini per dimostrare la robustezza agli elementi di disturbo.
"""
import cv2
import sys
import os
import numpy as np
from utils.robust_court_detection import robust_court_detection, enhanced_blue_court_mask, filter_central_court

def test_central_court_detection(image_path, save_results=True):
    """
    Esegue un test avanzato per rilevare specificamente il campo centrale, 
    ignorando le zone laterali.
    
    Args:
        image_path: Percorso dell'immagine da analizzare
        save_results: Se True, salva le immagini di risultato
    """
    if not os.path.exists(image_path):
        print(f"Errore: l'immagine {image_path} non esiste")
        return False
    
    # Carica l'immagine
    print(f"Caricamento dell'immagine {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Errore nel caricamento dell'immagine {image_path}")
        return False
    
    # Visualizza l'immagine originale
    cv2.imshow('Immagine originale', image)
    cv2.waitKey(1000)
    
    # Converti l'immagine in HSV per il rilevamento del colore
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1. Mostra prima la semplice maschera blu standard (per confronto)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    standard_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Applica morfologia alla maschera standard
    kernel = np.ones((7, 7), np.uint8)
    standard_mask = cv2.morphologyEx(standard_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Visualizza la maschera standard
    cv2.imshow('Maschera blu standard', standard_mask)
    cv2.waitKey(1000)
    
    # 2. Ora mostra la maschera migliorata per il campo centrale
    print("Applicazione del filtro avanzato per il campo centrale...")
    central_mask = enhanced_blue_court_mask(image, debug=True)
    
    # 3. Trova i contorni nella maschera centrale
    contours, _ = cv2.findContours(central_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Filtra i contorni per trovare il campo centrale
    filtered_contours = filter_central_court(contours, image.shape)
    
    # 5. Visualizza il risultato dei contorni filtrati
    if filtered_contours:
        central_contour = filtered_contours[0]
        filtered_result = image.copy()
        cv2.drawContours(filtered_result, [central_contour], -1, (0, 255, 0), 3)
        cv2.imshow('Campo centrale rilevato', filtered_result)
        cv2.waitKey(1000)
    else:
        print("Nessun contorno identificato come campo centrale")
    
    # 6. Esegui il rilevamento completo dei keypoints
    print("Rilevamento completo dei keypoints del campo...")
    keypoints = robust_court_detection(image, debug=True)
    
    # 7. Disegna i keypoints sull'immagine originale
    if keypoints:
        result_image = image.copy()
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(result_image, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.putText(
                result_image, 
                f"{i+1}", 
                (int(x) - 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        # Visualizza il risultato finale
        cv2.imshow('Risultato finale con keypoints', result_image)
        cv2.waitKey(0)
        
        # Salva i risultati
        if save_results:
            output_path = os.path.join(os.path.dirname(image_path), "central_court_detection_result.jpg")
            cv2.imwrite(output_path, result_image)
            print(f"Risultato salvato in {output_path}")
    else:
        print("Nessun keypoint rilevato")
    
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_central_court_detection(image_path)
    else:
        # Prova a usare l'immagine predefinita nella directory dei dati di input
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "data", "input", "frame.png")
        
        if os.path.exists(default_path):
            print(f"Uso dell'immagine predefinita: {default_path}")
            test_central_court_detection(default_path)
        else:
            print("Specificare il percorso dell'immagine come argomento, ad esempio:")
            print("python test_central_court.py data/input/frame.png")
