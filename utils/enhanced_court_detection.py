"""
Sistema avanzato e migliorato per il rilevamento dei keypoints nel campo da padel.
Questo modulo integra sia il rilevamento tradizionale che quello avanzato per ottenere
risultati più precisi nelle diverse condizioni di illuminazione e visibilità del campo.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from utils.court_detection import detect_court_keypoints_cv, improve_court_detection, combine_keypoints
from utils.advanced_court_detection import detect_court_keypoints_advanced

def detect_court_keypoints_enhanced(
    image: np.ndarray, 
    detection_method: str = "combined",
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Rileva i 12 keypoints del campo da padel utilizzando un approccio combinato
    che integra più metodi di rilevamento.
    
    Args:
        image: L'immagine del campo da padel (formato BGR)
        detection_method: Metodo di rilevamento ("traditional", "advanced", o "combined")
        debug: Se True, visualizza le immagini intermedie del processo
        
    Returns:
        Lista delle coordinate (x, y) dei 12 keypoints ordinati come nell'immagine di riferimento
    """
    # Copia l'immagine originale per il debug
    debug_img = image.copy() if debug else None
    height, width = image.shape[:2]
    keypoints = []
    
    # Fase 1: Rilevamento con il metodo tradizionale (basato sulle linee bianche)
    if detection_method in ["traditional", "combined"]:
        print("Rilevamento corte: utilizzo metodo tradizionale basato sulle linee bianche...")
        keypoints_cv = detect_court_keypoints_cv(image, debug=debug)
        
        if len(keypoints_cv) == 12:
            print("Rilevati tutti i 12 keypoints con metodo tradizionale")
            
            # Mostra i keypoints per debug se richiesto
            if debug:
                debug_img = image.copy()
                for i, kp in enumerate(keypoints_cv):
                    cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                    cv2.putText(
                        debug_img, 
                        f"k{i+1}", 
                        (int(kp[0]) + 10, int(kp[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )
                cv2.imshow('Keypoints Tradizionali', debug_img)
                cv2.waitKey(1000)
                cv2.imwrite('debug_cv_keypoints.jpg', debug_img)
            
            return keypoints_cv
        
        keypoints = keypoints_cv
        print(f"Rilevati {len(keypoints_cv)} keypoints con metodo tradizionale")
    
    # Fase 2: Rilevamento con il metodo avanzato (basato sul campo blu e linee bianche)
    if detection_method in ["advanced", "combined"]:
        print("Rilevamento corte: utilizzo metodo avanzato basato sulla maschera blu e sulle linee bianche...")
        keypoints_adv = detect_court_keypoints_advanced(image, debug=debug)
        
        # Mostra i keypoints avanzati per debug se richiesto
        if debug and keypoints_adv:
            debug_img = image.copy()
            for i, kp in enumerate(keypoints_adv):
                cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
                cv2.putText(
                    debug_img, 
                    f"k{i+1}", 
                    (int(kp[0]) + 10, int(kp[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
            cv2.imshow('Keypoints Avanzati', debug_img)
            cv2.waitKey(1000)
        
        if detection_method == "advanced":
            # Usa solo il metodo avanzato
            keypoints = keypoints_adv
            print(f"Rilevati {len(keypoints_adv)} keypoints con metodo avanzato (modalità esclusiva)")
        elif len(keypoints) == 0:
            # Non abbiamo keypoints dal metodo tradizionale, usiamo quelli avanzati
            keypoints = keypoints_adv
            print(f"Nessun keypoint dal metodo tradizionale, utilizzati {len(keypoints_adv)} keypoints dal metodo avanzato")
        else:
            # Combinazione dei due metodi
            print(f"Combinazione di {len(keypoints)} keypoints tradizionali con {len(keypoints_adv)} keypoints avanzati")
            combined_keypoints = []
            
            # Se entrambi hanno keypoints, combiniamo
            if len(keypoints) > 0 and len(keypoints_adv) > 0:
                # I keypoints_adv sono considerati come "avanzati", simili ai keypoints YOLO
                combined_keypoints = combine_keypoints(keypoints_adv, keypoints, image.shape)
                print(f"Combinazione completata: {len(combined_keypoints)} keypoints totali")
                
                # Mostra i keypoints combinati per debug se richiesto
                if debug:
                    debug_img = image.copy()
                    for i, kp in enumerate(combined_keypoints):
                        cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 5, (255, 255, 0), -1)
                        cv2.putText(
                            debug_img, 
                            f"k{i+1}", 
                            (int(kp[0]) + 10, int(kp[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (255, 255, 255), 
                            2
                        )
                    cv2.imshow('Keypoints Combinati', debug_img)
                    cv2.waitKey(1000)
                    cv2.imwrite('debug_combined_keypoints.jpg', debug_img)
            elif len(keypoints_adv) > 0:
                combined_keypoints = keypoints_adv
            else:
                combined_keypoints = keypoints
            
            keypoints = combined_keypoints
    
    # Fase 3: Verifica e correzione dei keypoints
    # Se abbiamo più di 12 keypoints dopo la combinazione, seleziona i migliori 12
    if len(keypoints) > 12:
        print(f"Rilevati {len(keypoints)} keypoints, selezione dei 12 migliori...")
        # Selezioniamo i punti più distanti tra loro per massimizzare la copertura
        keypoints = select_best_keypoints(keypoints, 12)
    
    # Se abbiamo ancora keypoints incompleti, proviamo a stimarli
    if 0 < len(keypoints) < 12:
        print(f"Tentativo di miglioramento: stima dei keypoints mancanti ({len(keypoints)}/12)...")
        keypoints = improve_court_detection(image, keypoints, debug=debug)
        print(f"Dopo miglioramento: {len(keypoints)}/12 keypoints")
    
    # Fase 4: Ordinamento e normalizzazione dei keypoints
    if len(keypoints) == 12:
        keypoints = order_keypoints(keypoints)
        keypoints = correct_keypoint_alignment(keypoints, image.shape[:2])
        
        # Visualizza i keypoints finali
        if debug:
            final_img = image.copy()
            for i, kp in enumerate(keypoints):
                cv2.circle(final_img, (int(kp[0]), int(kp[1])), 8, (0, 0, 255), -1)
                cv2.putText(
                    final_img, 
                    f"k{i+1}", 
                    (int(kp[0]) + 10, int(kp[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
            cv2.imshow('Keypoints Finali Ordinati', final_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("Rilevamento keypoints completato con successo!")
    
    return keypoints

def select_best_keypoints(
    keypoints: List[Tuple[float, float]], 
    n: int = 12
) -> List[Tuple[float, float]]:
    """
    Seleziona i migliori n keypoints dalla lista di keypoints disponibili.
    Utilizza una strategia che massimizza la distanza tra i punti selezionati.
    
    Args:
        keypoints: Lista di tutti i keypoints disponibili
        n: Numero di keypoints da selezionare
        
    Returns:
        Lista dei migliori n keypoints
    """
    if len(keypoints) <= n:
        return keypoints
    
    # Inizializza con il punto più a sinistra e più in alto
    selected = [min(keypoints, key=lambda p: p[0] + p[1])]
    
    while len(selected) < n:
        # Trova il punto che ha la massima distanza minima dai punti già selezionati
        max_min_dist = -1
        best_point = None
        
        for point in keypoints:
            if point in selected:
                continue
            
            # Calcola la distanza minima da tutti i punti già selezionati
            min_dist = min(math.sqrt((point[0] - s[0])**2 + (point[1] - s[1])**2) for s in selected)
            
            # Se questa distanza è maggiore di quella trovata finora
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_point = point
        
        if best_point:
            selected.append(best_point)
    
    return selected

def order_keypoints(keypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Ordina i keypoints secondo la disposizione standard del campo da padel.
    
    Disposizione standard:
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
    
    Args:
        keypoints: Lista dei keypoints da ordinare
        
    Returns:
        Lista di keypoints ordinati secondo la disposizione standard
    """
    if len(keypoints) != 12:
        return keypoints
    
    # Ordinamento per coordinata Y (dal basso verso l'alto)
    sorted_by_y = sorted(keypoints, key=lambda p: p[1], reverse=True)
    
    # Identificazione delle righe (5 righe, dall'alto verso il basso)
    # Assumiamo che le righe siano distribuite abbastanza uniformemente
    y_values = [p[1] for p in sorted_by_y]
    y_min, y_max = min(y_values), max(y_values)
    y_range = y_max - y_min
    
    # Crea 5 gruppi di y (dal basso verso l'alto)
    row_thresholds = [y_min + i * y_range / 4 for i in range(5)]
    
    # Dividi i keypoints per righe
    rows = [[] for _ in range(5)]
    
    for p in keypoints:
        for i in range(4):
            if row_thresholds[i] <= p[1] < row_thresholds[i+1]:
                rows[i].append(p)
                break
        else:  # Se il punto è sopra l'ultima soglia
            rows[4].append(p)
    
    # Ordina i punti in ogni riga per coordinata X (da sinistra a destra)
    for i in range(5):
        rows[i] = sorted(rows[i], key=lambda p: p[0])
    
    # Costruisce la lista ordinata finale
    ordered_keypoints = []
    
    # Riga 1 (dal basso): k1, k2
    if len(rows[0]) >= 2:
        ordered_keypoints.append(rows[0][0])  # k1
        ordered_keypoints.append(rows[0][-1])  # k2
    else:
        # Se non abbiamo abbastanza punti, aggiungi punti fittizi
        ordered_keypoints.extend([(0, 0)] * (2 - len(rows[0])))
    
    # Riga 2: k3, k4, k5
    if len(rows[1]) >= 3:
        ordered_keypoints.append(rows[1][0])  # k3
        ordered_keypoints.append(rows[1][len(rows[1]) // 2])  # k4
        ordered_keypoints.append(rows[1][-1])  # k5
    else:
        # Cerchiamo di fare del nostro meglio con ciò che abbiamo
        ordered_keypoints.append(rows[1][0] if len(rows[1]) > 0 else (0, 0))  # k3
        if len(rows[1]) >= 3:
            ordered_keypoints.append(rows[1][len(rows[1]) // 2])  # k4
        else:
            ordered_keypoints.append((0, 0))  # k4 fittizio
        ordered_keypoints.append(rows[1][-1] if len(rows[1]) > 1 else (0, 0))  # k5
    
    # Riga 3: k6, k7
    if len(rows[2]) >= 2:
        ordered_keypoints.append(rows[2][0])  # k6
        ordered_keypoints.append(rows[2][-1])  # k7
    else:
        ordered_keypoints.extend([(0, 0)] * (2 - len(rows[2])))
    
    # Riga 4: k8, k9, k10
    if len(rows[3]) >= 3:
        ordered_keypoints.append(rows[3][0])  # k8
        ordered_keypoints.append(rows[3][len(rows[3]) // 2])  # k9
        ordered_keypoints.append(rows[3][-1])  # k10
    else:
        ordered_keypoints.append(rows[3][0] if len(rows[3]) > 0 else (0, 0))  # k8
        if len(rows[3]) >= 3:
            ordered_keypoints.append(rows[3][len(rows[3]) // 2])  # k9
        else:
            ordered_keypoints.append((0, 0))  # k9 fittizio
        ordered_keypoints.append(rows[3][-1] if len(rows[3]) > 1 else (0, 0))  # k10
    
    # Riga 5 (dall'alto): k11, k12
    if len(rows[4]) >= 2:
        ordered_keypoints.append(rows[4][0])  # k11
        ordered_keypoints.append(rows[4][-1])  # k12
    else:
        ordered_keypoints.extend([(0, 0)] * (2 - len(rows[4])))
    
    return ordered_keypoints

def correct_keypoint_alignment(
    keypoints: List[Tuple[float, float]],
    image_shape: Tuple[int, int]
) -> List[Tuple[float, float]]:
    """
    Corregge l'allineamento dei keypoints per garantire linee dritte.
    
    Args:
        keypoints: Lista dei keypoints da correggere
        image_shape: Dimensioni dell'immagine (h, w)
        
    Returns:
        Lista di keypoints corretti
    """
    if len(keypoints) != 12:
        return keypoints
    
    corrected = keypoints.copy()
    
    # Identifica i keypoints nelle righe orizzontali
    rows = [
        [0, 1],      # k1, k2 (fondo)
        [2, 3, 4],   # k3, k4, k5 (linea servizio inferiore)
        [5, 6],      # k6, k7 (linea di metà campo)
        [7, 8, 9],   # k8, k9, k10 (linea servizio superiore)
        [10, 11]     # k11, k12 (cima)
    ]
    
    # Assicurati che i keypoints in ogni riga orizzontale abbiano la stessa coordinata y
    for row_indices in rows:
        valid_kps = [keypoints[i] for i in row_indices if keypoints[i] != (0, 0)]
        if len(valid_kps) > 1:  # Abbiamo almeno due punti validi
            avg_y = sum(kp[1] for kp in valid_kps) / len(valid_kps)
            for idx in row_indices:
                if keypoints[idx] != (0, 0):
                    x, _ = keypoints[idx]
                    corrected[idx] = (x, avg_y)
    
    # Identifica i keypoints nelle colonne verticali
    cols = [
        [0, 2, 5, 7, 10],   # Colonna sinistra
        [3, 8],             # Colonna centrale
        [1, 4, 6, 9, 11]    # Colonna destra
    ]
    
    # Assicurati che i keypoints in ogni colonna verticale abbiano la stessa coordinata x
    for col_indices in cols:
        valid_kps = [keypoints[i] for i in col_indices if keypoints[i] != (0, 0)]
        if len(valid_kps) > 1:  # Abbiamo almeno due punti validi
            avg_x = sum(kp[0] for kp in valid_kps) / len(valid_kps)
            for idx in col_indices:
                if keypoints[idx] != (0, 0):
                    _, y = keypoints[idx]
                    corrected[idx] = (avg_x, y)
    
    return corrected
