"""
Modulo specializzato per il rilevamento preciso del campo da padel.
Si concentra sull'estrazione robusta del perimetro del campo blu e delle linee bianche,
gestendo elementi di disturbo come ombre, riflessi o interruzioni nel colore del campo.
"""
import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Optional

def extract_court_boundary(
    image: np.ndarray,
    debug: bool = False
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Estrae con precisione il perimetro del campo da padel blu.
    
    Args:
        image: Immagine di input (formato BGR)
        debug: Se True, visualizza le immagini intermedie
        
    Returns:
        Tuple con:
        - Maschera binaria del campo
        - Lista dei punti del contorno del campo
    """
    # 1. Converti l'immagine in HSV per un miglior rilevamento del blu
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    
    # 2. Applica una doppia soglia per il colore blu, una più restrittiva e una più permissiva
    # Intervallo di blu più restrittivo (solo il blu intenso del campo)
    lower_blue1 = np.array([100, 80, 50])
    upper_blue1 = np.array([130, 255, 255])
    
    # Intervallo di blu più ampio (per catturare anche blu più chiaro o scuro)
    lower_blue2 = np.array([90, 50, 50])
    upper_blue2 = np.array([140, 255, 255])
    
    # Crea le due maschere
    strict_mask = cv2.inRange(hsv, lower_blue1, upper_blue1)
    broad_mask = cv2.inRange(hsv, lower_blue2, upper_blue2)
    
    # 3. Applica operazioni morfologiche per chiudere i buchi nella maschera restrittiva
    kernel = np.ones((7, 7), np.uint8)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Usa la maschera ampia per riempire i vuoti rimanenti
    combined_mask = cv2.bitwise_or(strict_mask, broad_mask)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    if debug:
        cv2.imshow('Maschera blu restrittiva', strict_mask)
        cv2.waitKey(500)
        cv2.imshow('Maschera blu combinata', combined_mask)
        cv2.waitKey(500)
    
    # 5. Trova i contorni nella maschera combinata
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Nessun contorno rilevato per il campo")
        return combined_mask, []
    
    # 6. Nuova implementazione: filtra i contorni in base alla posizione e dimensione
    # per trovare il campo centrale, ignorando le aree laterali
    filtered_contours = filter_central_court(contours, image.shape)
    
    if filtered_contours:
        # Usa il miglior contorno filtrato (campo centrale)
        court_contour = filtered_contours[0]
    else:
        # Fallback: usa il contorno più grande (comportamento precedente)
        print("Nessun contorno identificato come campo centrale, uso il più grande")
        court_contour = max(contours, key=cv2.contourArea)
    
    # 7. Applica la maschera avanzata specificamente per il campo centrale
    central_court_mask = enhanced_blue_court_mask(image, combined_mask, debug)
    
    # 8. Ritorna alla procedura standard: trova e filtra i contorni della maschera centrale
    central_contours, _ = cv2.findContours(central_court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if central_contours:
        # Aggiorna il contorno con quello dalla maschera migliorata
        court_contour = max(central_contours, key=cv2.contourArea)
    
    # 9. Approssima il contorno per ottenere il perimetro semplificato
    epsilon = 0.01 * cv2.arcLength(court_contour, True)
    approx_contour = cv2.approxPolyDP(court_contour, epsilon, True)
    
    # 10. Crea una maschera finale usando solo il contorno del campo
    refined_mask = np.zeros_like(strict_mask)
    cv2.drawContours(refined_mask, [approx_contour], -1, 255, -1)
    
    # 11. Usa il riempimento dei buchi per assicurarsi che il campo sia completamente coperto
    # Crea un'immagine di lavoro più grande per evitare problemi ai bordi
    h, w = refined_mask.shape[:2]
    floodfill_mask = np.zeros((h+2, w+2), np.uint8)
    refined_mask_copy = refined_mask.copy()
    cv2.floodFill(refined_mask_copy, floodfill_mask, (0, 0), 255)
    inverted_floodfill = cv2.bitwise_not(refined_mask_copy)
    final_mask = refined_mask | inverted_floodfill
    
    # 12. Visualizza il risultato finale per debug
    if debug:
        court_outline = image.copy()
        cv2.drawContours(court_outline, [approx_contour], -1, (0, 255, 0), 2)
        cv2.imshow('Contorno finale del campo centrale', court_outline)
        cv2.waitKey(500)
        
        cv2.imshow('Maschera finale del campo centrale', final_mask)
        cv2.waitKey(500)
    
    contour_points = [tuple(point[0]) for point in approx_contour]
    return final_mask, contour_points

def extract_court_lines(
    image: np.ndarray, 
    court_mask: np.ndarray,
    debug: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Estrae con precisione le linee bianche del campo da padel.
    
    Args:
        image: Immagine di input (formato BGR)
        court_mask: Maschera binaria del campo
        debug: Se True, visualizza le immagini intermedie
        
    Returns:
        Tuple con liste di linee (x1, y1, x2, y2):
        - Linee orizzontali
        - Linee verticali
    """
    # 1. Converti l'immagine in spazio colore che enfatizzi il bianco
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 2. Usa il canale V (luminosità) per rilevare le linee bianche
    # e applica una sogliatura adattiva per gestire meglio variazioni di illuminazione
    block_size = 21  # Deve essere dispari
    c_value = 5  # Costante sottratta dalla media
    
    # Applica la sogliatura adattiva solo all'interno della maschera del campo
    v_masked = cv2.bitwise_and(v, v, mask=court_mask)
    
    # Sogliatura adattativa per trovare le aree bianche
    white_binary = cv2.adaptiveThreshold(
        v_masked, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        block_size, 
        c_value
    )
    
    # 3. Applica filtri morfologici per migliorare il rilevamento delle linee
    kernel = np.ones((3, 3), np.uint8)
    white_binary = cv2.morphologyEx(white_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    white_binary = cv2.morphologyEx(white_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Rileva i bordi con Canny
    edges = cv2.Canny(white_binary, 50, 150, apertureSize=3)
    
    if debug:
        cv2.imshow('Linee bianche (soglia adattativa)', white_binary)
        cv2.waitKey(500)
        cv2.imshow('Bordi rilevati', edges)
        cv2.waitKey(500)
    
    # 5. Rileva le linee con la trasformata di Hough probabilistica
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,          # Minimo numero di intersezioni per rilevare una linea
        minLineLength=40,      # Lunghezza minima della linea
        maxLineGap=15          # Massima distanza tra segmenti da connettere
    )
    
    if lines is None:
        print("Nessuna linea rilevata")
        return [], []
    
    # 6. Separa le linee in orizzontali e verticali
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcola l'angolo per determinare se la linea è orizzontale o verticale
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        
        if (angle < 30) or (angle > 150):   # Linee orizzontali (entro 30° dall'orizzontale)
            horizontal_lines.append((x1, y1, x2, y2))
        elif (60 < angle < 120):           # Linee verticali (entro 30° dalla verticale)
            vertical_lines.append((x1, y1, x2, y2))
    
    # 7. Visualizza le linee rilevate per debug
    if debug:
        line_image = image.copy()
        
        # Disegna linee orizzontali
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Disegna linee verticali
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imshow('Linee rilevate', line_image)
        cv2.waitKey(500)
    
    # 8. Unisci linee simili
    horizontal_lines = merge_similar_lines(horizontal_lines)
    vertical_lines = merge_similar_lines(vertical_lines)
    
    # 9. Visualizza il risultato finale
    if debug:
        final_lines = image.copy()
        
        # Disegna linee orizzontali unite
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(final_lines, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Disegna linee verticali unite
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(final_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imshow('Linee finali', final_lines)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return horizontal_lines, vertical_lines

def merge_similar_lines(
    lines: List[Tuple[int, int, int, int]],
    distance_threshold: float = 20.0,
    angle_threshold: float = 5.0
) -> List[Tuple[int, int, int, int]]:
    """
    Unisce le linee simili in termini di posizione e orientamento.
    
    Args:
        lines: Lista di linee (x1, y1, x2, y2)
        distance_threshold: Distanza massima tra le linee da unire
        angle_threshold: Differenza massima di angolo tra le linee da unire
        
    Returns:
        Lista di linee unite
    """
    if not lines:
        return []
    
    # Calcola parametri per ogni linea
    line_params = []
    
    for x1, y1, x2, y2 in lines:
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        line_params.append({
            'coords': (x1, y1, x2, y2),
            'length': length,
            'angle': angle,
            'mid_point': (mid_x, mid_y)
        })
    
    # Ordina le linee per lunghezza (dalla più lunga alla più corta)
    line_params.sort(key=lambda x: x['length'], reverse=True)
    
    # Lista per le linee unite
    merged_lines = []
    used_indices = set()
    
    # Unisci le linee simili
    for i, line1 in enumerate(line_params):
        if i in used_indices:
            continue
        
        current_line = line1
        used_indices.add(i)
        
        for j, line2 in enumerate(line_params):
            if j in used_indices or j == i:
                continue
            
            # Controlla se le linee sono simili (angolo e distanza)
            angle_diff = abs(current_line['angle'] - line2['angle']) % 180
            angle_diff = min(angle_diff, 180 - angle_diff)
            
            if angle_diff > angle_threshold:
                continue
            
            # Calcola distanza tra i punti centrali
            mid1 = current_line['mid_point']
            mid2 = line2['mid_point']
            
            dist = math.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
            
            if dist > distance_threshold:
                continue
            
            # Le linee sono simili, uniscile
            x1_1, y1_1, x2_1, y2_1 = current_line['coords']
            x1_2, y1_2, x2_2, y2_2 = line2['coords']
            
            # Trova i punti estremi
            points = [(x1_1, y1_1), (x2_1, y2_1), (x1_2, y1_2), (x2_2, y2_2)]
            
            if abs(current_line['angle']) < 45 or abs(current_line['angle']) > 135:
                # Linea prevalentemente orizzontale, ordina per x
                points.sort(key=lambda p: p[0])
                x1, y1 = points[0]
                x2, y2 = points[-1]
            else:
                # Linea prevalentemente verticale, ordina per y
                points.sort(key=lambda p: p[1])
                x1, y1 = points[0]
                x2, y2 = points[-1]
            
            # Aggiorna la linea corrente
            current_line = {
                'coords': (x1, y1, x2, y2),
                'length': math.sqrt((x2 - x1)**2 + (y2 - y1)**2),
                'angle': math.atan2(y2 - y1, x2 - x1) * 180 / math.pi,
                'mid_point': ((x1 + x2) / 2, (y1 + y2) / 2)
            }
            
            used_indices.add(j)
        
        merged_lines.append(current_line['coords'])
    
    return merged_lines

def find_court_keypoints(
    image: np.ndarray,
    court_contour: List[Tuple[int, int]],
    horizontal_lines: List[Tuple[int, int, int, int]],
    vertical_lines: List[Tuple[int, int, int, int]],
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Rileva i 12 punti chiave del campo da padel combinando il contorno e le linee.
    
    Args:
        image: Immagine di input (formato BGR)
        court_contour: Lista dei punti che formano il contorno del campo
        horizontal_lines: Lista delle linee orizzontali (x1, y1, x2, y2)
        vertical_lines: Lista delle linee verticali (x1, y1, x2, y2)
        debug: Se True, visualizza le immagini intermedie
        
    Returns:
        Lista delle coordinate dei 12 keypoints
    """
    # 1. Trova le intersezioni tra linee orizzontali e verticali
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
                intersections.append(intersection)
    
    # 2. Filtra le intersezioni che sono troppo vicine tra loro
    filtered_intersections = filter_nearby_points(intersections, 20)
    
    # 3. Trova gli angoli del campo dal contorno
    corners = find_corners_from_contour(court_contour)
    
    # 4. Combina intersezioni e angoli, e organizza nei 12 keypoints standard
    all_points = filtered_intersections + corners
    keypoints = organize_court_keypoints(all_points, image.shape[:2])
    
    # 5. Visualizza i keypoints per debug
    if debug:
        debug_img = image.copy()
        
        # Disegna le intersezioni
        for i, (x, y) in enumerate(filtered_intersections):
            cv2.circle(debug_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(
                debug_img, 
                f"I{i}", 
                (int(x) - 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                1
            )
        
        # Disegna gli angoli
        for i, (x, y) in enumerate(corners):
            cv2.circle(debug_img, (int(x), int(y)), 7, (255, 0, 0), -1)
            cv2.putText(
                debug_img, 
                f"C{i}", 
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
        
        cv2.imshow('Intersezioni e angoli', debug_img)
        cv2.waitKey(500)
        
        # Disegna i keypoints finali
        if keypoints:
            keypoints_img = image.copy()
            for i, (x, y) in enumerate(keypoints):
                cv2.circle(keypoints_img, (int(x), int(y)), 8, (0, 255, 0), -1)
                cv2.putText(
                    keypoints_img, 
                    f"K{i+1}", 
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
            cv2.imshow('Keypoints finali', keypoints_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return keypoints

def line_intersection(
    line1_start: Tuple[float, float],
    line1_end: Tuple[float, float],
    line2_start: Tuple[float, float],
    line2_end: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """
    Calcola l'intersezione tra due linee definite da due punti ciascuna.
    
    Args:
        line1_start: Punto di inizio della prima linea (x, y)
        line1_end: Punto di fine della prima linea (x, y)
        line2_start: Punto di inizio della seconda linea (x, y)
        line2_end: Punto di fine della seconda linea (x, y)
        
    Returns:
        Punto di intersezione (x, y) o None se le linee non si intersecano
    """
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end
    
    # Calcola il denominatore
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Se il denominatore è zero, le linee sono parallele
    if abs(den) < 1e-6:
        return None
    
    # Calcola i numeratori per i parametri ua e ub
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
    
    # Se ua e ub sono tra 0 e 1, l'intersezione è all'interno dei segmenti
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # Calcola le coordinate del punto di intersezione
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    
    return None

def filter_nearby_points(
    points: List[Tuple[float, float]], 
    distance_threshold: float
) -> List[Tuple[float, float]]:
    """
    Filtra i punti che sono troppo vicini tra loro, mantenendo solo uno per gruppo.
    
    Args:
        points: Lista di punti (x, y)
        distance_threshold: Distanza minima tra due punti distinti
        
    Returns:
        Lista di punti filtrati
    """
    if not points:
        return []
    
    # Ordina i punti per la coordinata y, poi per x
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))
    
    # Raggruppa i punti vicini e mantieni solo il primo di ogni gruppo
    filtered_points = []
    
    while sorted_points:
        current = sorted_points.pop(0)
        filtered_points.append(current)
        
        # Rimuovi tutti i punti che sono vicini al punto corrente
        sorted_points = [
            p for p in sorted_points
            if math.sqrt((p[0] - current[0])**2 + (p[1] - current[1])**2) > distance_threshold
        ]
    
    return filtered_points

def find_corners_from_contour(contour: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """
    Trova i quattro angoli del campo dato il suo contorno.
    
    Args:
        contour: Lista di punti del contorno
        
    Returns:
        Lista dei quattro angoli del campo
    """
    if len(contour) < 4:
        return []
    
    # Trova il centro del campo
    xs, ys = zip(*contour)
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    
    # Classifica i punti per quadrante rispetto al centro
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None
    
    for point in contour:
        x, y = point
        
        # Calcola distanze dal centro
        dx = x - center_x
        dy = y - center_y
        
        # Classifica per quadrante
        if dx < 0:  # Lato sinistro
            if dy < 0:  # Quadrante in alto a sinistra
                if top_left is None or x + y < top_left[0] + top_left[1]:
                    top_left = point
            else:  # Quadrante in basso a sinistra
                if bottom_left is None or x - y < bottom_left[0] - bottom_left[1]:
                    bottom_left = point
        else:  # Lato destro
            if dy < 0:  # Quadrante in alto a destra
                if top_right is None or -x + y < -top_right[0] + top_right[1]:
                    top_right = point
            else:  # Quadrante in basso a destra
                if bottom_right is None or -x - y < -bottom_right[0] - bottom_right[1]:
                    bottom_right = point
    
    corners = []
    if bottom_left:
        corners.append(bottom_left)
    if bottom_right:
        corners.append(bottom_right)
    if top_left:
        corners.append(top_left)
    if top_right:
        corners.append(top_right)
    
    return corners

def organize_court_keypoints(
    points: List[Tuple[float, float]],
    image_shape: Tuple[int, int]
) -> List[Tuple[float, float]]:
    """
    Organizza i punti rilevati nei 12 keypoints standard del campo da padel.
    
    Args:
        points: Lista di tutti i punti rilevati
        image_shape: Dimensioni dell'immagine (height, width)
        
    Returns:
        Lista ordinata dei 12 keypoints
    """
    h, w = image_shape
    
    # Se abbiamo meno di 12 punti, probabilmente ne mancano alcuni
    if len(points) < 4:
        print(f"Troppo pochi punti rilevati ({len(points)}), impossibile determinare i keypoints")
        return []
    
    # Ordina i punti per coordinata Y (dall'alto verso il basso)
    points_by_y = sorted(points, key=lambda p: p[1])
    
    # Dividi i punti in 4-5 righe orizzontali
    num_rows = min(5, len(points) // 2)  # Almeno 2 punti per riga
    y_ranges = []
    
    if num_rows > 0:
        min_y = points_by_y[0][1]
        max_y = points_by_y[-1][1]
        y_step = (max_y - min_y) / num_rows
        
        for i in range(num_rows):
            lower = min_y + i * y_step
            upper = min_y + (i + 1) * y_step if i < num_rows - 1 else max_y + 1
            y_ranges.append((lower, upper))
    
    # Raggruppa i punti per riga
    rows = []
    
    for lower, upper in y_ranges:
        row = [p for p in points if lower <= p[1] < upper]
        if row:
            rows.append(sorted(row, key=lambda p: p[0]))  # Ordina per X
    
    # Prepara i 12 keypoints ordinati
    keypoints = [None] * 12
    
    # Se abbiamo 5 righe, possiamo assegnare i keypoints direttamente
    if len(rows) >= 5:
        # Prima riga (dall'alto): k11, k12
        if len(rows[0]) >= 2:
            keypoints[10] = rows[0][0]  # k11 (in alto a sinistra)
            keypoints[11] = rows[0][-1]  # k12 (in alto a destra)
        elif len(rows[0]) == 1:
            # Se c'è solo un punto, mettilo a k11
            keypoints[10] = rows[0][0]
        
        # Seconda riga: k8, k9, k10
        if len(rows[1]) >= 3:
            keypoints[7] = rows[1][0]  # k8 (sinistra)
            keypoints[8] = rows[1][len(rows[1]) // 2]  # k9 (centro)
            keypoints[9] = rows[1][-1]  # k10 (destra)
        elif len(rows[1]) == 2:
            keypoints[7] = rows[1][0]  # k8 (sinistra)
            keypoints[9] = rows[1][-1]  # k10 (destra)
        
        # Terza riga: k6, k7
        if len(rows[2]) >= 2:
            keypoints[5] = rows[2][0]  # k6 (sinistra)
            keypoints[6] = rows[2][-1]  # k7 (destra)
        
        # Quarta riga: k3, k4, k5
        if len(rows[3]) >= 3:
            keypoints[2] = rows[3][0]  # k3 (sinistra)
            keypoints[3] = rows[3][len(rows[3]) // 2]  # k4 (centro)
            keypoints[4] = rows[3][-1]  # k5 (destra)
        elif len(rows[3]) == 2:
            keypoints[2] = rows[3][0]  # k3 (sinistra)
            keypoints[4] = rows[3][-1]  # k5 (destra)
        
        # Quinta riga (dal basso): k1, k2
        if len(rows[4]) >= 2:
            keypoints[0] = rows[4][0]  # k1 (in basso a sinistra)
            keypoints[1] = rows[4][-1]  # k2 (in basso a destra)
        elif len(rows[4]) == 1:
            # Se c'è solo un punto, mettilo a k1
            keypoints[0] = rows[4][0]
    else:
        # Non abbiamo abbastanza righe, proviamo con un altro approccio
        corners = find_corners_from_points(points)
        
        if len(corners) >= 4:
            # Assegna gli angoli del campo
            keypoints[0] = corners[0]  # k1 (in basso a sinistra)
            keypoints[1] = corners[1]  # k2 (in basso a destra)
            keypoints[10] = corners[2]  # k11 (in alto a sinistra)
            keypoints[11] = corners[3]  # k12 (in alto a destra)
            
            # Trova i punti interni
            internal_points = [p for p in points if p not in corners]
            
            # Stima i keypoints restanti
            keypoints = estimate_missing_keypoints(keypoints, internal_points, image_shape)
    
    # Filtra i keypoints None e sostituisci con punti stimati
    if keypoints.count(None) > 0:
        keypoints = estimate_missing_keypoints(keypoints, points, image_shape)
    
    # Assicurati che abbiamo esattamente 12 keypoints
    valid_keypoints = [kp for kp in keypoints if kp is not None]
    
    return valid_keypoints

def find_corners_from_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Trova i quattro angoli tra un insieme di punti.
    
    Args:
        points: Lista di punti
        
    Returns:
        Lista dei quattro angoli (in basso a sinistra, in basso a destra, in alto a sinistra, in alto a destra)
    """
    if len(points) < 4:
        return points
    
    # Calcola il bounding box
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Cerca i punti più vicini agli angoli del bounding box
    corners = [None] * 4  # [bottom-left, bottom-right, top-left, top-right]
    corner_coords = [(min_x, max_y), (max_x, max_y), (min_x, min_y), (max_x, min_y)]
    
    for i, (corner_x, corner_y) in enumerate(corner_coords):
        best_dist = float('inf')
        best_point = None
        
        for p in points:
            dist = (p[0] - corner_x)**2 + (p[1] - corner_y)**2
            if dist < best_dist:
                best_dist = dist
                best_point = p
        
        corners[i] = best_point
    
    return corners

def estimate_missing_keypoints(
    keypoints: List[Optional[Tuple[float, float]]],
    points: List[Tuple[float, float]],
    image_shape: Tuple[int, int]
) -> List[Tuple[float, float]]:
    """
    Stima i keypoints mancanti basandosi su quelli esistenti.
    
    Args:
        keypoints: Lista di keypoints (con possibili valori None)
        points: Lista di tutti i punti rilevati
        image_shape: Dimensioni dell'immagine (height, width)
        
    Returns:
        Lista completa di 12 keypoints
    """
    h, w = image_shape
    result = list(keypoints)  # Copia i keypoints esistenti
    
    # Se abbiamo gli angoli del campo (k1, k2, k11, k12), possiamo stimare gli altri
    corners_indices = [0, 1, 10, 11]  # Indici dei keypoints che rappresentano gli angoli
    corners = [keypoints[i] for i in corners_indices]
    
    if all(corners):
        # Abbiamo tutti e quattro gli angoli
        bottom_left, bottom_right, top_left, top_right = corners
        
        # Stima k3 e k5 (1/4 dal basso)
        if result[2] is None:
            x = bottom_left[0]
            y = bottom_left[1] + (top_left[1] - bottom_left[1]) * 0.25
            result[2] = (x, y)
        
        if result[4] is None:
            x = bottom_right[0]
            y = bottom_right[1] + (top_right[1] - bottom_right[1]) * 0.25
            result[4] = (x, y)
        
        # Stima k6 e k7 (metà campo)
        if result[5] is None:
            x = bottom_left[0]
            y = bottom_left[1] + (top_left[1] - bottom_left[1]) * 0.5
            result[5] = (x, y)
        
        if result[6] is None:
            x = bottom_right[0]
            y = bottom_right[1] + (top_right[1] - bottom_right[1]) * 0.5
            result[6] = (x, y)
        
        # Stima k8 e k10 (3/4 dal basso)
        if result[7] is None:
            x = bottom_left[0]
            y = bottom_left[1] + (top_left[1] - bottom_left[1]) * 0.75
            result[7] = (x, y)
        
        if result[9] is None:
            x = bottom_right[0]
            y = bottom_right[1] + (top_right[1] - bottom_right[1]) * 0.75
            result[9] = (x, y)
        
        # Stima k4 e k9 (linea centrale)
        if result[3] is None and result[2] is not None and result[4] is not None:
            x = (result[2][0] + result[4][0]) / 2
            y = (result[2][1] + result[4][1]) / 2
            result[3] = (x, y)
        
        if result[8] is None and result[7] is not None and result[9] is not None:
            x = (result[7][0] + result[9][0]) / 2
            y = (result[7][1] + result[9][1]) / 2
            result[8] = (x, y)
    
    # Verifica se ci sono ancora keypoints mancanti dopo la stima
    if None in result:
        # In questo caso, possiamo provare a cercare tra i punti disponibili
        for i, kp in enumerate(result):
            if kp is None:
                # Trova il punto più vicino alla posizione stimata
                estimated_x = w / 2  # Default al centro dell'immagine
                estimated_y = h / 2
                
                # Stima basata sulla posizione relativa nel campo
                if i == 0:  # k1 (in basso a sinistra)
                    estimated_x = w * 0.1
                    estimated_y = h * 0.9
                elif i == 1:  # k2 (in basso a destra)
                    estimated_x = w * 0.9
                    estimated_y = h * 0.9
                elif i == 2:  # k3
                    estimated_x = w * 0.1
                    estimated_y = h * 0.7
                elif i == 3:  # k4
                    estimated_x = w * 0.5
                    estimated_y = h * 0.7
                elif i == 4:  # k5
                    estimated_x = w * 0.9
                    estimated_y = h * 0.7
                elif i == 5:  # k6
                    estimated_x = w * 0.1
                    estimated_y = h * 0.5
                elif i == 6:  # k7
                    estimated_x = w * 0.9
                    estimated_y = h * 0.5
                elif i == 7:  # k8
                    estimated_x = w * 0.1
                    estimated_y = h * 0.3
                elif i == 8:  # k9
                    estimated_x = w * 0.5
                    estimated_y = h * 0.3
                elif i == 9:  # k10
                    estimated_x = w * 0.9
                    estimated_y = h * 0.3
                elif i == 10:  # k11 (in alto a sinistra)
                    estimated_x = w * 0.1
                    estimated_y = h * 0.1
                elif i == 11:  # k12 (in alto a destra)
                    estimated_x = w * 0.9
                    estimated_y = h * 0.1
                
                # Cerca il punto più vicino alla posizione stimata
                best_dist = float('inf')
                best_point = None
                
                for p in points:
                    dist = (p[0] - estimated_x)**2 + (p[1] - estimated_y)**2
                    if dist < best_dist and p not in result:
                        best_dist = dist
                        best_point = p
                
                if best_point:
                    result[i] = best_point
                else:
                    # Se non troviamo un punto adatto, usiamo la stima
                    result[i] = (estimated_x, estimated_y)
    
    return result

def robust_court_detection(
    image: np.ndarray,
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Funzione principale per il rilevamento robusto del campo e dei suoi keypoints.
    
    Args:
        image: Immagine di input (formato BGR)
        debug: Se True, visualizza le immagini intermedie
        
    Returns:
        Lista dei 12 keypoints del campo
    """
    # 1. Estrai il perimetro del campo
    court_mask, court_contour = extract_court_boundary(image, debug)
    
    # 2. Estrai le linee bianche del campo
    horizontal_lines, vertical_lines = extract_court_lines(image, court_mask, debug)
    
    # 3. Trova i keypoints combinando contorno e linee
    keypoints = find_court_keypoints(image, court_contour, horizontal_lines, vertical_lines, debug)
    
    # 4. Assicurati che i keypoints siano allineati correttamente
    if len(keypoints) == 12:
        keypoints = align_keypoints(keypoints)
    
    return keypoints

def align_keypoints(
    keypoints: List[Tuple[float, float]], 
    tolerance: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Allinea i keypoints per garantire che formino una griglia regolare.
    
    Args:
        keypoints: Lista dei 12 keypoints
        tolerance: Tolleranza per l'allineamento (come frazione della dimensione del campo)
        
    Returns:
        Lista dei keypoints allineati
    """
    if len(keypoints) != 12:
        return keypoints
    
    result = list(keypoints)
    
    # Calcola la dimensione del campo
    min_x = min(kp[0] for kp in keypoints)
    max_x = max(kp[0] for kp in keypoints)
    min_y = min(kp[1] for kp in keypoints)
    max_y = max(kp[1] for kp in keypoints)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Soglie di tolleranza
    x_threshold = width * tolerance
    y_threshold = height * tolerance
    
    # Definisci le righe orizzontali e le colonne verticali
    horizontal_rows = [
        [0, 1],      # k1, k2
        [2, 3, 4],   # k3, k4, k5
        [5, 6],      # k6, k7
        [7, 8, 9],   # k8, k9, k10
        [10, 11]     # k11, k12
    ]
    
    vertical_cols = [
        [0, 2, 5, 7, 10],   # Colonna sinistra
        [3, 8],             # Colonna centrale
        [1, 4, 6, 9, 11]    # Colonna destra
    ]
    
    # Allinea le righe orizzontali (stessa coordinata y)
    for row in horizontal_rows:
        y_values = [keypoints[i][1] for i in row]
        avg_y = sum(y_values) / len(y_values)
        
        for i in row:
            # Se la coordinata y è vicina alla media, allineala
            if abs(keypoints[i][1] - avg_y) < y_threshold:
                result[i] = (result[i][0], avg_y)
    
    # Allinea le colonne verticali (stessa coordinata x)
    for col in vertical_cols:
        x_values = [keypoints[i][0] for i in col]
        avg_x = sum(x_values) / len(x_values)
        
        for i in col:
            # Se la coordinata x è vicina alla media, allineala
            if abs(keypoints[i][0] - avg_x) < x_threshold:
                result[i] = (avg_x, result[i][1])
    
    return result

# Funzione principale per testare il rilevamento del campo
def test_court_detection(image_path: str, debug: bool = True) -> None:
    """
    Testa il rilevamento del campo su un'immagine.
    
    Args:
        image_path: Percorso dell'immagine
        debug: Se True, visualizza le immagini intermedie
    """
    import os
    if not os.path.exists(image_path):
        print(f"Errore: l'immagine {image_path} non esiste")
        return
    
    # Carica l'immagine
    image = cv2.imread(image_path)
    if image is None:
        print(f"Errore nel caricamento dell'immagine {image_path}")
        return
    
    # Rileva il campo
    keypoints = robust_court_detection(image, debug)
    
    if keypoints:
        print(f"Rilevati {len(keypoints)} keypoints")
        
        # Disegna i keypoints sull'immagine
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
        
        # Salva l'immagine risultante
        output_path = "court_detection_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"Immagine risultante salvata in {output_path}")
    else:
        print("Nessun keypoint rilevato")

# Esegui il test se il file viene eseguito direttamente
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_court_detection(sys.argv[1])
    else:
        print("Specificare il percorso dell'immagine come argomento")
