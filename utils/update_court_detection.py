"""
Funzioni aggiuntive per migliorare il rilevamento dei keypoints del campo da padel.
Questo modulo estende le funzionalità del modulo court_detection.py con algoritmi
più robusti e specifici per il rilevamento delle linee verticali e la stima dei keypoints.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import math
import os

def enhance_vertical_line_detection(
    edges: np.ndarray,
    white_mask: np.ndarray,
    debug: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    Funzione specializzata per il miglioramento del rilevamento delle linee verticali.
    Implementa tecniche specifiche per enfatizzare e rilevare le linee verticali del campo.
    
    Args:
        edges: Mappa dei bordi dall'immagine (e.g., output di Canny)
        white_mask: Maschera binaria delle aree bianche
        debug: Se True, mostra immagini intermedie per il debug
        
    Returns:
        Lista di linee verticali nel formato (x1, y1, x2, y2)
    """
    # Crea una copia degli edges per la manipolazione
    vertical_edges = edges.copy()
    
    # Applica operazioni morfologiche per enfatizzare le strutture verticali
    # Definiamo un kernel verticale alto e stretto
    kernel_vertical = np.ones((15, 1), np.uint8)
    
    # Operazione di chiusura per connettere gap verticali
    vertical_edges = cv2.morphologyEx(vertical_edges, cv2.MORPH_CLOSE, kernel_vertical, iterations=1)
    
    # Dilatazione specifica per linee verticali
    vertical_edges = cv2.dilate(vertical_edges, kernel_vertical, iterations=1)
    
    if debug:
        cv2.imshow('Enhanced Vertical Edges', vertical_edges)
        cv2.waitKey(500)
    
    # Trasformata di Hough specifica per linee verticali
    # Parametri più sensibili per linee verticali
    vertical_lines = cv2.HoughLinesP(
        vertical_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=40,            # Soglia più bassa per linee verticali
        minLineLength=80,       # Lunghezza minima minore per catturare più linee
        maxLineGap=30           # Gap maggiore per connettere segmenti frammentati
    )
    
    if vertical_lines is None:
        print("Nessuna linea verticale rilevata con parametri specializzati.")
        return []
    
    # Filtra linee per mantenere solo quelle verticali
    filtered_vertical_lines = []
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcola angolo rispetto all'asse orizzontale
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        
        # Mantieni solo linee con angolo vicino a 90 gradi
        if 70 <= angle <= 110:
            # Verifica che la linea passi attraverso pixel bianchi
            num_samples = 15
            points_x = np.linspace(x1, x2, num_samples).astype(int)
            points_y = np.linspace(y1, y2, num_samples).astype(int)
            
            white_points = 0
            for x, y in zip(points_x, points_y):
                if 0 <= y < white_mask.shape[0] and 0 <= x < white_mask.shape[1]:
                    if white_mask[y, x] > 0:
                        white_points += 1
            
            if white_points / num_samples >= 0.6:  # Soglia più bassa per linee verticali
                filtered_vertical_lines.append((x1, y1, x2, y2))
    
    if debug and filtered_vertical_lines:
        height, width = edges.shape
        v_line_img = np.zeros((height, width, 3), dtype=np.uint8)
        for x1, y1, x2, y2 in filtered_vertical_lines:
            cv2.line(v_line_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.imshow('Specialized Vertical Lines', v_line_img)
        cv2.waitKey(500)
    
    print(f"Rilevate {len(filtered_vertical_lines)} linee verticali specializzate")
    return filtered_vertical_lines

def refine_keypoint_positions(
    keypoints: List[Tuple[float, float]],
    white_mask: np.ndarray,
    radius: int = 10
) -> List[Tuple[float, float]]:
    """
    Raffina la posizione dei keypoints cercando l'intersezione più precisa 
    nelle vicinanze di ogni keypoint.
    
    Args:
        keypoints: Lista di keypoints da raffinare
        white_mask: Maschera binaria delle aree bianche
        radius: Raggio di ricerca attorno a ciascun keypoint
        
    Returns:
        Lista di keypoints con posizioni raffinate
    """
    refined_keypoints = []
    
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        
        # Estrai una ROI intorno al keypoint
        x_min = max(0, x - radius)
        y_min = max(0, y - radius)
        x_max = min(white_mask.shape[1] - 1, x + radius)
        y_max = min(white_mask.shape[0] - 1, y + radius)
        
        roi = white_mask[y_min:y_max+1, x_min:x_max+1]
        if roi.size == 0:
            refined_keypoints.append(kp)
            continue
        
        # Cerca il centro di massa dei pixel bianchi nella ROI
        # che potrebbe rappresentare meglio l'intersezione
        y_coords, x_coords = np.nonzero(roi)
        
        if len(y_coords) > 0:
            # Calcola il centro di massa
            center_x = x_min + np.mean(x_coords)
            center_y = y_min + np.mean(y_coords)
            refined_keypoints.append((center_x, center_y))
        else:
            # Se non ci sono pixel bianchi nella ROI, mantieni il keypoint originale
            refined_keypoints.append(kp)
    
    return refined_keypoints

def verify_keypoint_structure(
    keypoints: List[Tuple[float, float]],
    image_shape: Tuple[int, int, int]
) -> bool:
    """
    Verifica che la struttura dei keypoints sia valida geometricamente.
    Controlla proporzioni, allineamenti e distanze relative.
    
    Args:
        keypoints: Lista di keypoints da verificare
        image_shape: Dimensioni dell'immagine (h, w, c)
        
    Returns:
        True se la struttura è valida, False altrimenti
    """
    if len(keypoints) != 12:
        print("Numero di keypoints non valido.")
        return False
    
    # Verifica le proporzioni del campo
    # I keypoints sono ordinati seguendo la convenzione 1-12
    try:
        # Estrai le coordinate
        k1, k2 = keypoints[0], keypoints[1]  # Bottom left, bottom right
        k11, k12 = keypoints[10], keypoints[11]  # Top left, top right
        
        # Calcola la larghezza e l'altezza del campo
        width = math.sqrt((k2[0] - k1[0])**2 + (k2[1] - k1[1])**2)
        height = math.sqrt((k11[0] - k1[0])**2 + (k11[1] - k1[1])**2)
        
        # Verifica il rapporto larghezza/altezza (dovrebbe essere circa 2:1 per un campo da padel)
        ratio = width / height if height > 0 else 0
        if not (1.8 <= ratio <= 2.2):
            print(f"Rapporto larghezza/altezza non valido: {ratio}")
            return False
        
        # Verifica l'allineamento verticale dei keypoints sulla stessa linea
        left_x_values = [keypoints[0][0], keypoints[2][0], keypoints[5][0], keypoints[7][0], keypoints[10][0]]
        right_x_values = [keypoints[1][0], keypoints[4][0], keypoints[6][0], keypoints[9][0], keypoints[11][0]]
        center_x_values = [keypoints[3][0], keypoints[8][0]]
        
        # Calcola la variazione delle coordinate x per ciascun gruppo
        left_x_std = np.std(left_x_values)
        right_x_std = np.std(right_x_values)
        
        # Le linee verticali dovrebbero avere una deviazione standard bassa
        if left_x_std > width * 0.1 or right_x_std > width * 0.1:
            print(f"Allineamento verticale non valido. STD sx: {left_x_std}, STD dx: {right_x_std}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Errore durante la verifica della struttura: {str(e)}")
        return False

def get_court_polygon(keypoints: List[Tuple[float, float]]) -> np.ndarray:
    """
    Estrae il poligono del campo dai keypoints.
    
    Args:
        keypoints: Lista dei 12 keypoints del campo
        
    Returns:
        Array numpy con le coordinate del poligono (perimetro del campo)
    """
    if len(keypoints) != 12:
        return np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        
    # Estrai i keypoints degli angoli del campo (k1, k2, k11, k12)
    k1 = keypoints[0]  # Bottom left
    k2 = keypoints[1]  # Bottom right
    k11 = keypoints[10]  # Top left
    k12 = keypoints[11]  # Top right
    
    # Crea il poligono
    polygon = np.array([k1, k2, k12, k11])
    
    return polygon
