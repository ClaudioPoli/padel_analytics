#!/usr/bin/env python
"""
Script di test per il rilevamento robusto del campo da padel.
Questo script consente di testare l'algoritmo su un'immagine specifica e
visualizzare i risultati passo dopo passo.
"""
import cv2
import sys
import os
import numpy as np
from utils.robust_court_detection import robust_court_detection

def test_court_detection(image_path, save_results=True):
    """
    Esegue il test del rilevamento del campo e visualizza i risultati.
    
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
    
    # Ridimensiona l'immagine se è troppo grande per la visualizzazione
    max_display_dim = 1200
    h, w = image.shape[:2]
    scale = 1.0
    
    if max(h, w) > max_display_dim:
        scale = max_display_dim / max(h, w)
        display_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        print(f"Immagine ridimensionata per la visualizzazione (scala: {scale:.2f})")
    else:
        display_img = image.copy()
    
    # Visualizza l'immagine originale
    cv2.imshow('Immagine originale', display_img)
    cv2.waitKey(1000)
    
    # Rileva il campo con il debug attivato
    print("Rilevamento robusto del campo in corso...")
    keypoints = robust_court_detection(image, debug=True)
    
    if not keypoints:
        print("Errore: nessun keypoint rilevato")
        return False
    
    print(f"Rilevati {len(keypoints)} keypoints")
    
    # Disegna i keypoints sull'immagine originale
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
    result_display = cv2.resize(result_image, (int(w * scale), int(h * scale))) if scale < 1.0 else result_image.copy()
    cv2.imshow('Risultato finale', result_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Salva i risultati
    if save_results:
        output_path = os.path.join(os.path.dirname(image_path), "court_detection_result.jpg")
        cv2.imwrite(output_path, result_image)
        print(f"Risultato salvato in {output_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_court_detection(image_path)
    else:
        # Prova a usare l'immagine predefinita nella directory dei dati di input
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "data", "input", "frame.png")
        
        if os.path.exists(default_path):
            print(f"Uso dell'immagine predefinita: {default_path}")
            test_court_detection(default_path)
        else:
            print("Specificare il percorso dell'immagine come argomento, ad esempio:")
            print("python test_court_detection.py data/input/frame.png")
