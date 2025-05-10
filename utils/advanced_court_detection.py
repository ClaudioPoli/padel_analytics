"""
Algoritmo avanzato per il rilevamento dei keypoints nel campo da padel.
Questa implementazione si concentra sul rilevamento accurato dei keypoints esattamente
dove si troverebbero in una selezione manuale: sulle linee bianche, nelle loro intersezioni
e sui limiti del campo (blu vs non-blu).
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

def detect_court_keypoints_advanced(
    image: np.ndarray, 
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Rileva i 12 keypoints del campo da padel concentrandosi su:
    1. Linee bianche e loro intersezioni
    2. Limiti del campo (confine tra area blu e non-blu)
    3. Angoli del campo
    
    Args:
        image: L'immagine del campo da padel (formato BGR)
        debug: Se True, visualizza le immagini intermedie del processo
        
    Returns:
        Lista delle coordinate (x, y) dei 12 keypoints ordinati
    """
    # Copia l'immagine originale per il debug
    debug_img = image.copy() if debug else None
    height, width = image.shape[:2]
    
    # FASE 1: Rilevamento del campo (area blu)
    # Converte in HSV per un migliore rilevamento del colore blu
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Intervallo per il blu tipico dei campi da padel (può richiedere calibrazione)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Crea una maschera per il campo blu
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Rimuovi il rumore e migliora la maschera
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    if debug:
        cv2.imshow('Campo Blu Rilevato', blue_mask)
        cv2.waitKey(500)
    
    # FASE 2: Rileva i contorni del campo blu
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Impossibile rilevare il campo blu.")
        return []
    
    # Prendi il contorno più grande (dovrebbe essere il campo)
    court_contour = max(contours, key=cv2.contourArea)
    
    # Approssima il contorno per ottenere i punti del poligono
    epsilon = 0.02 * cv2.arcLength(court_contour, True)
    approx_poly = cv2.approxPolyDP(court_contour, epsilon, True)
    
    # Disegna il contorno rilevato per il debug
    if debug:
        contour_img = image.copy()
        cv2.drawContours(contour_img, [approx_poly], 0, (0, 255, 0), 3)
        cv2.imshow('Contorno del Campo', contour_img)
        cv2.waitKey(500)
    
    # FASE 3: Rileva le linee bianche all'interno del campo
    # Crea una maschera per le linee bianche
    _, _, v = cv2.split(hsv)
    white_threshold = 180  # Valore alto per isolare il bianco
    _, white_mask = cv2.threshold(v, white_threshold, 255, cv2.THRESH_BINARY)
    
    # Applica la maschera del campo per rimuovere linee bianche esterne
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=blue_mask)
    
    # Migliora la maschera delle linee bianche
    white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    if debug:
        cv2.imshow('Linee Bianche', white_mask)
        cv2.waitKey(500)
    
    # FASE 4: Rileva i bordi e le linee
    edges = cv2.Canny(white_mask, 50, 150)
    
    # Applica la trasformata di Hough per rilevare le linee
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=50, 
        maxLineGap=30
    )
    
    if lines is None or len(lines) < 4:
        print("Linee insufficienti rilevate, rilasso i parametri...")
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30, 
            minLineLength=30, 
            maxLineGap=40
        )
    
    if lines is None:
        print("Impossibile rilevare le linee del campo.")
        return []
    
    # Estrai le linee
    extracted_lines = [line[0] for line in lines]
    
    # Dividi le linee in orizzontali e verticali
    horizontal_lines = []
    vertical_lines = []
    
    for line in extracted_lines:
        x1, y1, x2, y2 = line
        # Calcola l'angolo per classificare la linea
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        
        if (angle < 30) or (angle > 150):
            horizontal_lines.append(line)
        elif (angle > 60) and (angle < 120):
            vertical_lines.append(line)
    
    # FASE 5: Visualizza le linee per debug
    if debug:
        line_img = image.copy()
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.imshow('Linee Rilevate', line_img)
        cv2.waitKey(500)
    
    # FASE 6: Trova i punti di intersezione tra le linee
    intersections = []
    
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            x1_h, y1_h, x2_h, y2_h = h_line
            x1_v, y1_v, x2_v, y2_v = v_line
            
            # Calcola l'intersezione
            intersection = line_intersection(
                (x1_h, y1_h), (x2_h, y2_h), 
                (x1_v, y1_v), (x2_v, y2_v)
            )
            
            if intersection:
                x, y = intersection
                # Verifica che l'intersezione sia all'interno dell'immagine e del campo
                if (0 <= x < width) and (0 <= y < height) and point_in_mask(blue_mask, x, y):
                    intersections.append(intersection)
    
    # Filtra intersezioni troppo vicine tra loro
    filtered_intersections = filter_nearby_points(intersections, distance_threshold=20)
    
    # FASE 7: Identifica i punti degli angoli del campo utilizzando il contorno
    corner_points = extract_corner_points(approx_poly, blue_mask)
    
    # FASE 8: Visualizza le intersezioni e gli angoli
    if debug and filtered_intersections:
        intersection_img = image.copy()
        for i, (x, y) in enumerate(filtered_intersections):
            cv2.circle(intersection_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(
                intersection_img, 
                f"I{i}", 
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        # Disegna gli angoli del campo
        for i, (x, y) in enumerate(corner_points):
            cv2.circle(intersection_img, (int(x), int(y)), 7, (255, 0, 0), -1)
            cv2.putText(
                intersection_img, 
                f"C{i}", 
                (int(x) - 15, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        cv2.imshow('Intersezioni e Angoli', intersection_img)
        cv2.waitKey(500)
    
    # FASE 9: Combina intersezioni e angoli, poi organizza nei 12 keypoints standard
    all_points = filtered_intersections + corner_points
    keypoints = organize_court_keypoints(all_points, blue_mask, horizontal_lines, vertical_lines)
    
    # FASE 10: Se mancano keypoints, stimali dalla geometria
    if len(keypoints) < 12:
        keypoints = estimate_missing_court_keypoints(keypoints, blue_mask.shape[:2])
    
    # Verifica finale: assicurati che ci siano tutti i 12 keypoints
    if len(keypoints) == 12:
        # Visualizza i keypoints finali
        if debug:
            final_img = image.copy()
            for i, (x, y) in enumerate(keypoints):
                cv2.circle(final_img, (int(x), int(y)), 8, (0, 0, 255), -1)
                cv2.putText(
                    final_img, 
                    f"k{i+1}", 
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
            cv2.imshow('Keypoints Finali', final_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"Rilevati con successo tutti i 12 keypoints!")
        return keypoints
    else:
        print(f"Rilevati solo {len(keypoints)} keypoints su 12 necessari.")
        return keypoints

def line_intersection(
    line1_start: Tuple[float, float], 
    line1_end: Tuple[float, float],
    line2_start: Tuple[float, float], 
    line2_end: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """Calcola il punto di intersezione tra due segmenti di linea."""
    # Converte i punti in vettori 2D
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end
    
    # Calcola i determinanti
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Linee parallele
    if abs(den) < 1e-6:
        return None
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
    
    # Verifica se l'intersezione è all'interno di entrambi i segmenti
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    
    return None

def filter_nearby_points(
    points: List[Tuple[float, float]], 
    distance_threshold: float
) -> List[Tuple[float, float]]:
    """Filtra i punti troppo vicini tra loro, mantenendo un punto per ogni gruppo."""
    if not points:
        return []
    
    result = []
    points = sorted(points, key=lambda p: (p[1], p[0]))  # Ordina per y, poi per x
    
    while points:
        current = points.pop(0)
        result.append(current)
        
        # Rimuovi tutti i punti vicini al punto corrente
        points = [
            p for p in points 
            if euclidean_distance(current, p) > distance_threshold
        ]
    
    return result

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcola la distanza euclidea tra due punti."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def extract_corner_points(approx_poly: np.ndarray, blue_mask: np.ndarray) -> List[Tuple[float, float]]:
    """
    Estrae i punti degli angoli del campo dal contorno approssimato.
    Seleziona i 4 punti più esterni che rappresentano gli angoli del campo.
    """
    # Trova i punti estremi del contorno (alto, basso, sinistra, destra)
    points = approx_poly.reshape(-1, 2)
    
    # Ordina per coordinate
    points_by_x = sorted(points, key=lambda p: p[0])
    points_by_y = sorted(points, key=lambda p: p[1])
    
    # Prendi i punti estremi
    left_points = points_by_x[:2]
    right_points = points_by_x[-2:]
    top_points = points_by_y[:2]
    bottom_points = points_by_y[-2:]
    
    # Combina i punti estremi e rimuovi i duplicati
    extreme_points = []
    for p in left_points + right_points + top_points + bottom_points:
        if not any(np.array_equal(p, ep) for ep in extreme_points):
            extreme_points.append(p)
    
    # Seleziona i 4 angoli principali
    corners = []
    if len(extreme_points) >= 4:
        # Calcola il centro del campo
        center_x = np.mean([p[0] for p in extreme_points])
        center_y = np.mean([p[1] for p in extreme_points])
        
        # Trova gli angoli nei quattro quadranti
        top_left = None
        top_right = None
        bottom_left = None
        bottom_right = None
        
        for p in extreme_points:
            x, y = p
            
            # Verifica che il punto sia all'interno della maschera blu
            if point_in_mask(blue_mask, x, y, margin=5):
                if x <= center_x and y <= center_y:  # Top-left
                    if top_left is None or (x + y) < (top_left[0] + top_left[1]):
                        top_left = (x, y)
                elif x > center_x and y <= center_y:  # Top-right
                    if top_right is None or (center_x*2 - x + y) < (center_x*2 - top_right[0] + top_right[1]):
                        top_right = (x, y)
                elif x <= center_x and y > center_y:  # Bottom-left
                    if bottom_left is None or (x + center_y*2 - y) < (bottom_left[0] + center_y*2 - bottom_left[1]):
                        bottom_left = (x, y)
                elif x > center_x and y > center_y:  # Bottom-right
                    if bottom_right is None or (center_x*2 - x + center_y*2 - y) < (center_x*2 - bottom_right[0] + center_y*2 - bottom_right[1]):
                        bottom_right = (x, y)
        
        # Aggiungi gli angoli in ordine: basso-sx, basso-dx, alto-sx, alto-dx
        if bottom_left is not None:
            corners.append(tuple(bottom_left))
        if bottom_right is not None:
            corners.append(tuple(bottom_right))
        if top_left is not None:
            corners.append(tuple(top_left))
        if top_right is not None:
            corners.append(tuple(top_right))
    
    # Se non abbiamo abbastanza angoli, cerca di stimarli diversamente
    if len(corners) < 4:
        # Trova il rettangolo che racchiude la maschera blu
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(blue_mask))
        
        # Usa i vertici del rettangolo come angoli
        corners = [(x, y+h), (x+w, y+h), (x, y), (x+w, y)]
    
    return corners

def point_in_mask(mask: np.ndarray, x: float, y: float, margin: int = 0) -> bool:
    """Verifica se un punto si trova all'interno di una maschera binaria."""
    h, w = mask.shape[:2]
    x, y = int(x), int(y)
    
    # Verifica i limiti dell'immagine
    if x < margin or y < margin or x >= w - margin or y >= h - margin:
        return False
    
    # Verifica se il punto è nella maschera
    return mask[y, x] > 0

def organize_court_keypoints(
    points: List[Tuple[float, float]], 
    blue_mask: np.ndarray,
    horizontal_lines: List,
    vertical_lines: List
) -> List[Tuple[float, float]]:
    """
    Organizza i punti rilevati nei 12 keypoints standard del campo da padel.
    
    La disposizione dei keypoints è la seguente:
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
    # Ordina i punti per coordinata Y (dall'alto verso il basso)
    sorted_points = sorted(points, key=lambda p: p[1])
    
    # Inizializza la lista dei 12 keypoints
    keypoints = [None] * 12
    
    # Se abbiamo almeno 12 punti, possiamo provare a determinare i keypoints
    if len(sorted_points) >= 12:
        # Raggruppa i punti per righe (basate sulla coordinata Y)
        h, w = blue_mask.shape[:2]
        y_tolerance = h * 0.08  # Tolleranza per punti sulla stessa riga
        
        rows = []
        current_row = [sorted_points[0]]
        
        for i in range(1, len(sorted_points)):
            if abs(sorted_points[i][1] - sorted_points[i-1][1]) <= y_tolerance:
                # Stesso livello Y, aggiungi alla riga corrente
                current_row.append(sorted_points[i])
            else:
                # Nuova riga
                rows.append(sorted(current_row, key=lambda p: p[0]))  # Ordina per X
                current_row = [sorted_points[i]]
        
        # Aggiungi l'ultima riga
        if current_row:
            rows.append(sorted(current_row, key=lambda p: p[0]))
        
        # Identifica le 5 righe standard di keypoints
        if len(rows) >= 5:
            # Seleziona le 5 righe più significative (quelle con più punti)
            # e ordinale dall'alto verso il basso
            significant_rows = sorted(rows, key=lambda row: row[0][1] if row else float('inf'))
            
            # Prendiamo fino a 5 righe
            significant_rows = significant_rows[:5]
            
            # Prima riga (dall'alto): k11, k12
            if len(significant_rows) >= 5 and len(significant_rows[0]) >= 2:
                keypoints[10] = significant_rows[0][0]  # k11 (in alto a sinistra)
                keypoints[11] = significant_rows[0][-1]  # k12 (in alto a destra)
            
            # Seconda riga: k8, k9, k10
            if len(significant_rows) >= 5 and len(significant_rows[1]) >= 2:
                keypoints[7] = significant_rows[1][0]  # k8 (sinistra)
                if len(significant_rows[1]) >= 3:
                    keypoints[8] = significant_rows[1][len(significant_rows[1]) // 2]  # k9 (centro)
                keypoints[9] = significant_rows[1][-1]  # k10 (destra)
            
            # Terza riga: k6, k7
            if len(significant_rows) >= 5 and len(significant_rows[2]) >= 2:
                keypoints[5] = significant_rows[2][0]  # k6 (sinistra)
                keypoints[6] = significant_rows[2][-1]  # k7 (destra)
            
            # Quarta riga: k3, k4, k5
            if len(significant_rows) >= 5 and len(significant_rows[3]) >= 2:
                keypoints[2] = significant_rows[3][0]  # k3 (sinistra)
                if len(significant_rows[3]) >= 3:
                    keypoints[3] = significant_rows[3][len(significant_rows[3]) // 2]  # k4 (centro)
                keypoints[4] = significant_rows[3][-1]  # k5 (destra)
            
            # Quinta riga: k1, k2
            if len(significant_rows) >= 5 and len(significant_rows[4]) >= 2:
                keypoints[0] = significant_rows[4][0]  # k1 (in basso a sinistra)
                keypoints[1] = significant_rows[4][-1]  # k2 (in basso a destra)
        elif len(rows) == 4:
            # Abbiamo una riga mancante, determiniamo quale in base alla loro distribuzione
            # e alla posizione relativa rispetto alla maschera blu
            
            # Prima riga (dall'alto): k11, k12
            if len(rows[0]) >= 2:
                keypoints[10] = rows[0][0]  # k11 (in alto a sinistra)
                keypoints[11] = rows[0][-1]  # k12 (in alto a destra)
            
            # Ultima riga (dal basso): k1, k2
            if len(rows[-1]) >= 2:
                keypoints[0] = rows[-1][0]  # k1 (in basso a sinistra)
                keypoints[1] = rows[-1][-1]  # k2 (in basso a destra)
            
            # Determiniamo quale riga manca in base alla distribuzione delle coordinata Y
            if len(rows) == 4:
                y_dists = [rows[i+1][0][1] - rows[i][0][1] for i in range(len(rows)-1)]
                max_gap_idx = y_dists.index(max(y_dists))
                
                # Mappatura in base a dove si trova il gap più grande
                if max_gap_idx == 0:  # Manca la seconda riga (k8, k9, k10)
                    if len(rows[1]) >= 2:
                        keypoints[5] = rows[1][0]  # k6 (sinistra)
                        keypoints[6] = rows[1][-1]  # k7 (destra)
                    
                    if len(rows[2]) >= 2:
                        keypoints[2] = rows[2][0]  # k3 (sinistra)
                        if len(rows[2]) >= 3:
                            keypoints[3] = rows[2][len(rows[2]) // 2]  # k4 (centro)
                        keypoints[4] = rows[2][-1]  # k5 (destra)
                elif max_gap_idx == 1:  # Manca la terza riga (k6, k7)
                    if len(rows[1]) >= 2:
                        keypoints[7] = rows[1][0]  # k8 (sinistra)
                        if len(rows[1]) >= 3:
                            keypoints[8] = rows[1][len(rows[1]) // 2]  # k9 (centro)
                        keypoints[9] = rows[1][-1]  # k10 (destra)
                    
                    if len(rows[2]) >= 2:
                        keypoints[2] = rows[2][0]  # k3 (sinistra)
                        if len(rows[2]) >= 3:
                            keypoints[3] = rows[2][len(rows[2]) // 2]  # k4 (centro)
                        keypoints[4] = rows[2][-1]  # k5 (destra)
                elif max_gap_idx == 2:  # Manca la quarta riga (k3, k4, k5)
                    if len(rows[1]) >= 2:
                        keypoints[7] = rows[1][0]  # k8 (sinistra)
                        if len(rows[1]) >= 3:
                            keypoints[8] = rows[1][len(rows[1]) // 2]  # k9 (centro)
                        keypoints[9] = rows[1][-1]  # k10 (destra)
                    
                    if len(rows[2]) >= 2:
                        keypoints[5] = rows[2][0]  # k6 (sinistra)
                        keypoints[6] = rows[2][-1]  # k7 (destra)
    
    # Se non abbiamo abbastanza righe, proviamo a derivare i keypoints dagli angoli
    if keypoints.count(None) > 6 and len(points) >= 4:
        # Trova i punti agli estremi (angoli)
        points_by_x = sorted(points, key=lambda p: p[0])
        points_by_y = sorted(points, key=lambda p: p[1])
        
        # Identifica i 4 angoli del campo
        bottom_left = None
        bottom_right = None
        top_left = None
        top_right = None
        
        for p in points:
            x, y = p
            if x < points_by_x[len(points) // 2][0]:  # Metà sinistra
                if y > points_by_y[len(points) // 2][1]:  # Metà inferiore
                    if bottom_left is None or (x + y) > (bottom_left[0] + bottom_left[1]):
                        bottom_left = (x, y)
                else:  # Metà superiore
                    if top_left is None or (x - y) < (top_left[0] - top_left[1]):
                        top_left = (x, y)
            else:  # Metà destra
                if y > points_by_y[len(points) // 2][1]:  # Metà inferiore
                    if bottom_right is None or (y - x) > (bottom_right[1] - bottom_right[0]):
                        bottom_right = (x, y)
                else:  # Metà superiore
                    if top_right is None or (x + y) > (top_right[0] + top_right[1]):
                        top_right = (x, y)
        
        # Assegna gli angoli ai keypoints corrispondenti
        if bottom_left:
            keypoints[0] = bottom_left  # k1
        if bottom_right:
            keypoints[1] = bottom_right  # k2
        if top_left:
            keypoints[10] = top_left  # k11
        if top_right:
            keypoints[11] = top_right  # k12
    
    # Filtra solo i keypoints validi (non None)
    valid_keypoints = [kp for kp in keypoints if kp is not None]
    
    return valid_keypoints

def estimate_missing_court_keypoints(
    partial_keypoints: List[Tuple[float, float]], 
    image_shape: Tuple[int, int]
) -> List[Tuple[float, float]]:
    """
    Stima i keypoints mancanti basandosi sui keypoints rilevati e
    sulla geometria standard del campo da padel.
    """
    h, w = image_shape
    
    # Crea array per i keypoints stimati (possiamo avere fino a 12 keypoints)
    keypoints = [(0, 0)] * 12
    
    # Copia i keypoints parziali nella struttura completa
    for i, kp in enumerate(partial_keypoints):
        if i < len(keypoints):
            keypoints[i] = kp
    
    # Se abbiamo almeno i 4 angoli principali (k1, k2, k11, k12), possiamo stimare gli altri
    if all(keypoints[i] != (0, 0) for i in [0, 1, 10, 11]):
        # k1 (in basso a sx), k2 (in basso a dx), k11 (in alto a sx), k12 (in alto a dx)
        
        # Stima k3, k5 (1/4 dal basso)
        if keypoints[2] == (0, 0):
            x1, y1 = keypoints[0]  # k1
            x11, y11 = keypoints[10]  # k11
            keypoints[2] = (x1, y1 + (y11 - y1) * 0.25)  # k3 a 1/4 tra k1 e k11
        
        if keypoints[4] == (0, 0):
            x2, y2 = keypoints[1]  # k2
            x12, y12 = keypoints[11]  # k12
            keypoints[4] = (x2, y2 + (y12 - y2) * 0.25)  # k5 a 1/4 tra k2 e k12
        
        # Stima k6, k7 (metà campo)
        if keypoints[5] == (0, 0):
            x1, y1 = keypoints[0]  # k1
            x11, y11 = keypoints[10]  # k11
            keypoints[5] = (x1, y1 + (y11 - y1) * 0.5)  # k6 a metà tra k1 e k11
        
        if keypoints[6] == (0, 0):
            x2, y2 = keypoints[1]  # k2
            x12, y12 = keypoints[11]  # k12
            keypoints[6] = (x2, y2 + (y12 - y2) * 0.5)  # k7 a metà tra k2 e k12
        
        # Stima k8, k10 (3/4 dall'alto)
        if keypoints[7] == (0, 0):
            x1, y1 = keypoints[0]  # k1
            x11, y11 = keypoints[10]  # k11
            keypoints[7] = (x1, y1 + (y11 - y1) * 0.75)  # k8 a 3/4 tra k1 e k11
        
        if keypoints[9] == (0, 0):
            x2, y2 = keypoints[1]  # k2
            x12, y12 = keypoints[11]  # k12
            keypoints[9] = (x2, y2 + (y12 - y2) * 0.75)  # k10 a 3/4 tra k2 e k12
        
        # Stima k4, k9 (punti centrali)
        if keypoints[3] == (0, 0) and keypoints[2] != (0, 0) and keypoints[4] != (0, 0):
            x3, y3 = keypoints[2]  # k3
            x5, y5 = keypoints[4]  # k5
            keypoints[3] = ((x3 + x5) / 2, (y3 + y5) / 2)  # k4 al centro tra k3 e k5
        
        if keypoints[8] == (0, 0) and keypoints[7] != (0, 0) and keypoints[9] != (0, 0):
            x8, y8 = keypoints[7]  # k8
            x10, y10 = keypoints[9]  # k10
            keypoints[8] = ((x8 + x10) / 2, (y8 + y10) / 2)  # k9 al centro tra k8 e k10
    
    # Casi più complessi con meno keypoints disponibili
    elif sum(1 for kp in keypoints if kp != (0, 0)) >= 4:
        # Identifica quali keypoints abbiamo
        valid_indices = [i for i, kp in enumerate(keypoints) if kp != (0, 0)]
        
        # Prova a identificare gli angoli mancanti
        top_left_idx = None
        top_right_idx = None
        bottom_left_idx = None
        bottom_right_idx = None
        
        valid_keypoints = [keypoints[i] for i in valid_indices]
        if valid_keypoints:
            # Trova i punti più estremi
            leftmost = min(valid_keypoints, key=lambda p: p[0])
            rightmost = max(valid_keypoints, key=lambda p: p[0])
            topmost = min(valid_keypoints, key=lambda p: p[1])
            bottommost = max(valid_keypoints, key=lambda p: p[1])
            
            # Identifica gli angoli per la loro posizione
            for i, kp in enumerate(keypoints):
                if kp != (0, 0):
                    # In alto a sinistra
                    if kp[0] <= (leftmost[0] + (rightmost[0] - leftmost[0]) * 0.25) and kp[1] <= (topmost[1] + (bottommost[1] - topmost[1]) * 0.25):
                        top_left_idx = i
                    
                    # In alto a destra
                    elif kp[0] >= (leftmost[0] + (rightmost[0] - leftmost[0]) * 0.75) and kp[1] <= (topmost[1] + (bottommost[1] - topmost[1]) * 0.25):
                        top_right_idx = i
                    
                    # In basso a sinistra
                    elif kp[0] <= (leftmost[0] + (rightmost[0] - leftmost[0]) * 0.25) and kp[1] >= (topmost[1] + (bottommost[1] - topmost[1]) * 0.75):
                        bottom_left_idx = i
                    
                    # In basso a destra
                    elif kp[0] >= (leftmost[0] + (rightmost[0] - leftmost[0]) * 0.75) and kp[1] >= (topmost[1] + (bottommost[1] - topmost[1]) * 0.75):
                        bottom_right_idx = i
            
            # Se abbiamo identificato almeno alcuni angoli, possiamo stimate gli altri
            if top_left_idx is not None and top_left_idx != 10:
                keypoints[10] = keypoints[top_left_idx]
            if top_right_idx is not None and top_right_idx != 11:
                keypoints[11] = keypoints[top_right_idx]
            if bottom_left_idx is not None and bottom_left_idx != 0:
                keypoints[0] = keypoints[bottom_left_idx]
            if bottom_right_idx is not None and bottom_right_idx != 1:
                keypoints[1] = keypoints[bottom_right_idx]
            
            # Stima gli angoli mancanti se possibile
            if keypoints[0] == (0, 0) and keypoints[10] != (0, 0) and keypoints[1] != (0, 0):
                x10, y10 = keypoints[10]
                x1, y1 = keypoints[1]
                keypoints[0] = (x10, y1)
            
            if keypoints[1] == (0, 0) and keypoints[11] != (0, 0) and keypoints[0] != (0, 0):
                x11, y11 = keypoints[11]
                x0, y0 = keypoints[0]
                keypoints[1] = (x11, y0)
            
            if keypoints[10] == (0, 0) and keypoints[0] != (0, 0) and keypoints[11] != (0, 0):
                x0, y0 = keypoints[0]
                x11, y11 = keypoints[11]
                keypoints[10] = (x0, y11)
            
            if keypoints[11] == (0, 0) and keypoints[1] != (0, 0) and keypoints[10] != (0, 0):
                x1, y1 = keypoints[1]
                x10, y10 = keypoints[10]
                keypoints[11] = (x1, y10)
    
    # Verifica che tutte le coordinate siano all'interno dell'immagine
    valid_keypoints = []
    for kp in keypoints:
        x, y = kp
        if kp != (0, 0) and 0 <= x < w and 0 <= y < h:
            valid_keypoints.append(kp)
    
    return valid_keypoints

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
        row_kps = [keypoints[i] for i in row_indices]
        if all(kp != (0, 0) for kp in row_kps) and len(row_kps) > 1:
            avg_y = sum(kp[1] for kp in row_kps) / len(row_kps)
            for idx in row_indices:
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
        col_kps = [keypoints[i] for i in col_indices]
        if all(kp != (0, 0) for kp in col_kps) and len(col_kps) > 1:
            avg_x = sum(kp[0] for kp in col_kps) / len(col_kps)
            for idx in col_indices:
                _, y = keypoints[idx]
                corrected[idx] = (avg_x, y)
    
    return corrected
