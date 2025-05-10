"""
Metodi per il rilevamento automatico dei keypoints del campo da padel.
Questo modulo include algoritmi basati sia sull'elaborazione tradizionale delle immagini
che sul rilevamento tramite YOLO, con capacità di stima per keypoints mancanti.
Supporta anche l'integrazione con metodi avanzati di rilevamento basati sulla
combinazione di analisi delle aree blu e delle linee bianche.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import math

def detect_court_keypoints_cv(
    image: np.ndarray, 
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Rileva i 12 keypoints del campo da padel utilizzando tecniche di elaborazione delle immagini.
    
    Si concentra SOLO sulle linee BIANCHE che delimitano il campo, ignorando altre linee.
    Utilizza rilevamento dei bordi, trasformata di Hough e analisi delle intersezioni
    per identificare i keypoints. Quando alcune linee non sono visibili, cerca di
    stimare i keypoints mancanti in base alla geometria del campo.
    
    Args:
        image: L'immagine del campo da padel (formato BGR di OpenCV)
        debug: Se True, mostra immagini intermedie per il debug
        
    Returns:
        Lista delle coordinate (x, y) dei 12 keypoints ordinati
    """
    # Copia l'immagine originale per il debug
    debug_img = image.copy() if debug else None
    
    # 1. Preparazione dell'immagine - Enfatizza le linee bianche
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Estrai solo il canale V (luminosità) dall'immagine HSV
    _, _, v = cv2.split(hsv)
    
    # Applica una sogliatura per isolare i pixel bianchi/molto chiari
    # Adatta la soglia in base al tipo di campo (valore di default è 200)
    white_threshold = 170
    _, white_mask = cv2.threshold(v, white_threshold, 255, cv2.THRESH_BINARY)
    
    if debug:
        cv2.imshow('White Mask', white_mask)
        cv2.waitKey(500)
    
    # Applica un filtro di sfocatura per ridurre il rumore
    white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)
    
    # Applica morfologia per chiudere piccoli gap nelle linee
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    if debug:
        cv2.imshow('Processed White Mask', white_mask)
        cv2.waitKey(500)
    
    # 2. Rilevamento dei bordi SOLO sulle aree bianche
    edges = cv2.Canny(white_mask, 50, 150)
    
    # Dilatazione per connettere linee frammentate
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    if debug:
        cv2.imshow('Edges', edges)
        cv2.waitKey(500)
    
    # 3. Trasformata di Hough per trovare le linee
    # Utilizziamo parametri più restrittivi per trovare solo le linee più rilevanti
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,            # Valore più alto = meno linee
        minLineLength=100,       # Lunghezza minima maggiore per filtrare linee corte
        maxLineGap=20            # Distanza massima tra segmenti da unire
    )
    
    if lines is None or len(lines) < 4:  # Abbiamo bisogno di almeno 4 linee per stimare i keypoints
        print("Rilevamento linee insufficiente. Provo a rilassare i parametri...")
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30, 
            minLineLength=80, 
            maxLineGap=30
        )
    
    if lines is None or len(lines) < 4:
        print("Impossibile rilevare abbastanza linee per i keypoints.")
        return []
    
    # Filtro aggiuntivo: mantieni solo le linee che passano attraverso pixel bianchi
    # nell'immagine originale (linee del campo)
    filtered_lines = []
    if lines is not None:
        for line in lines:
            # Estrai le coordinate della linea
            coords = line[0]
            x1, y1, x2, y2 = coords
            
            # Calcola punti equidistanti lungo la linea per verificare il colore
            num_samples = 10
            points_x = np.linspace(x1, x2, num_samples).astype(int)
            points_y = np.linspace(y1, y2, num_samples).astype(int)
            
            # Verifica che almeno 70% dei punti abbia colore bianco nella white_mask
            white_points = 0
            for x, y in zip(points_x, points_y):
                # Assicurati che i punti siano all'interno dell'immagine
                if 0 <= y < white_mask.shape[0] and 0 <= x < white_mask.shape[1]:
                    if white_mask[y, x] > 0:
                        white_points += 1
            
            if white_points / num_samples >= 0.7:  # Almeno 70% dei punti sono bianchi
                filtered_lines.append(coords)
    
    lines = filtered_lines  # Usa solo le linee filtrate
    
    print(f"Rilevate {len(lines)} linee bianche del campo")
    
    # Limita il numero di linee per migliorare le prestazioni
    if len(lines) > 100:
        lines = filter_and_reduce_lines(lines, white_mask, max_lines=100)
        print(f"Ridotte a {len(lines)} linee più significative")
    
    # Disegna le linee rilevate per il debug
    if debug and lines:
        line_img = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Detected White Lines', line_img)
        cv2.waitKey(500)
    
    # 4. Raggruppa le linee per orientamento (orizzontale/verticale)
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        # Calcola l'angolo per determinare se è orizzontale o verticale
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        
        if (angle < 30) or (angle > 150):  # Linee orizzontali (+/- 30 gradi)
            horizontal_lines.append(line)
        elif (angle > 60) and (angle < 120):  # Linee verticali (+/- 30 gradi dalla verticale)
            vertical_lines.append(line)
    
    print(f"Rilevate {len(horizontal_lines)} linee orizzontali e {len(vertical_lines)} linee verticali")
    
    # Se abbiamo poche linee verticali, utilizziamo un algoritmo specializzato
    if len(vertical_lines) < 5:
        from utils.update_court_detection import enhance_vertical_line_detection
        print("Attivazione rilevamento specializzato per linee verticali...")
        enhanced_vertical_lines = enhance_vertical_line_detection(edges, white_mask, debug)
        # Aggiungi le nuove linee verticali rilevate
        vertical_lines.extend(enhanced_vertical_lines)
        print(f"Dopo miglioramento: {len(vertical_lines)} linee verticali")
    
    # Disegna le linee categorizzate per il debug
    if debug:
        h_line_img = np.zeros_like(image)
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(h_line_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imshow('Horizontal Lines', h_line_img)
        cv2.waitKey(500)
        
        v_line_img = np.zeros_like(image)
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(v_line_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.imshow('Vertical Lines', v_line_img)
        cv2.waitKey(500)
    
    # 5. Trova le intersezioni tra linee orizzontali e verticali
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
                # Verifica che l'intersezione sia all'interno dell'immagine
                if (0 <= x < image.shape[1]) and (0 <= y < image.shape[0]):
                    intersections.append(intersection)
    
    # 6. Filtra e raggruppa le intersezioni simili (potrebbero esserci duplicati)
    filtered_intersections = filter_nearby_points(intersections, distance_threshold=20)
    
    # Aggiungiamo un raffinamento della posizione dei keypoints in base ai pixel bianchi
    if filtered_intersections:
        from utils.update_court_detection import refine_keypoint_positions
        filtered_intersections = refine_keypoint_positions(filtered_intersections, white_mask)
    
    if debug and filtered_intersections:
        intersection_img = image.copy()
        for i, (x, y) in enumerate(filtered_intersections):
            cv2.circle(intersection_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(
                intersection_img, 
                f"{i}", 
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        cv2.imshow('Intersections', intersection_img)
        cv2.waitKey(500)
    
    # 7. Organizzare le intersezioni nei 12 keypoints del campo
    keypoints = organize_keypoints(filtered_intersections, image.shape)
    
    # 8. Stima i keypoints mancanti se necessario
    if len(keypoints) < 12:
        keypoints = estimate_missing_keypoints(keypoints, image.shape)
    
    # Visualizza i keypoints finali per il debug
    if debug and keypoints:
        keypoint_img = image.copy()
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(keypoint_img, (int(x), int(y)), 8, (0, 0, 255), -1)
            cv2.putText(
                keypoint_img, 
                f"k{i+1}", 
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        cv2.imshow('Final Keypoints', keypoint_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
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
    if den == 0:
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

def organize_keypoints(
    intersections: List[Tuple[float, float]], 
    image_shape: Tuple[int, int, int]
) -> List[Tuple[float, float]]:
    """
    Organizza le intersezioni rilevate nei 12 keypoints del campo da padel.
    
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
    
    Args:
        intersections: Lista delle intersezioni rilevate
        image_shape: Dimensioni dell'immagine (h, w, c)
    
    Returns:
        Lista dei 12 keypoints ordinati
    """
    if not intersections:
        return []
    
    h, w = image_shape[:2]
    keypoints = [None] * 12  # Lista per i 12 keypoints
    
    # Ordina le intersezioni per coordinata y (dall'alto verso il basso)
    sorted_by_y = sorted(intersections, key=lambda p: p[1])
    
    # Analizziamo la distribuzione delle coordinate y per trovare le righe
    y_values = [p[1] for p in sorted_by_y]
    
    # Se abbiamo troppe intersezioni, proviamo a identificare cluster naturali
    # Utilizziamo un approccio più sofisticato per raggruppare le y in base a righe reali
    if len(intersections) > 12:
        # Calcola le differenze tra coordinate y consecutive
        y_diffs = [y_values[i+1] - y_values[i] for i in range(len(y_values)-1)]
        
        # Cerca gap significativi per identificare le righe (differenze più grandi della media)
        if y_diffs:
            mean_diff = sum(y_diffs) / len(y_diffs)
            significant_gaps = [i for i, diff in enumerate(y_diffs) if diff > mean_diff * 1.5]
            
            # Usa questi gap per dividere in righe
            row_indices = [0] + [i+1 for i in significant_gaps] + [len(sorted_by_y)]
            
            rows = []
            for i in range(len(row_indices)-1):
                start_idx = row_indices[i]
                end_idx = row_indices[i+1]
                # Ordina i punti in questa riga per x
                row_points = sorted(sorted_by_y[start_idx:end_idx], key=lambda p: p[0])
                rows.append(row_points)
        else:
            # Fallback se non ci sono abbastanza differenze
            rows = []
    else:
        # Se abbiamo poche intersezioni, possiamo processarle direttamente
        prev_y = -1
        current_row = []
        rows = []
        y_tolerance = h * 0.08  # Tolleranza per punti sulla stessa riga
        
        for point in sorted_by_y:
            if prev_y == -1 or abs(point[1] - prev_y) <= y_tolerance:
                current_row.append(point)
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda p: p[0]))
                    current_row = [point]
            prev_y = point[1]
        
        if current_row:
            rows.append(sorted(current_row, key=lambda p: p[0]))
    
    # Filtra le righe: se abbiamo più di 5 righe, vogliamo mantenere le più significative
    if len(rows) > 5:
        # Ordina le righe per numero di punti (privilegiando quelle con più punti)
        rows = sorted(rows, key=lambda r: len(r), reverse=True)
        # Prendi le 5 righe con più punti
        rows = rows[:5]
        # Riordina per coordinata y
        rows = sorted(rows, key=lambda r: r[0][1] if r else float('inf'))
    
    # Con le righe ora definite, mappiamo i keypoints
    if len(rows) == 5:  # Scenario ideale: abbiamo tutte le 5 righe
        # Prima riga (dall'alto): k11, k12
        top_row = rows[0]
        if len(top_row) >= 2:
            keypoints[10] = top_row[0]  # k11 (sinistra)
            keypoints[11] = top_row[-1]  # k12 (destra)
        
        # Seconda riga: k8, k9, k10
        second_row = rows[1]
        if len(second_row) >= 3:
            keypoints[7] = second_row[0]  # k8 (sinistra)
            keypoints[8] = second_row[len(second_row) // 2]  # k9 (centro)
            keypoints[9] = second_row[-1]  # k10 (destra)
        elif len(second_row) == 2:
            keypoints[7] = second_row[0]  # k8 (sinistra)
            keypoints[9] = second_row[-1]  # k10 (destra)
        
        # Terza riga: k6, k7
        third_row = rows[2]
        if len(third_row) >= 2:
            keypoints[5] = third_row[0]  # k6 (sinistra)
            keypoints[6] = third_row[-1]  # k7 (destra)
        
        # Quarta riga: k3, k4, k5
        fourth_row = rows[3]
        if len(fourth_row) >= 3:
            keypoints[2] = fourth_row[0]  # k3 (sinistra)
            keypoints[3] = fourth_row[len(fourth_row) // 2]  # k4 (centro)
            keypoints[4] = fourth_row[-1]  # k5 (destra)
        elif len(fourth_row) == 2:
            keypoints[2] = fourth_row[0]  # k3 (sinistra)
            keypoints[4] = fourth_row[-1]  # k5 (destra)
        
        # Quinta riga (più in basso): k1, k2
        bottom_row = rows[4]
        if len(bottom_row) >= 2:
            keypoints[0] = bottom_row[0]  # k1 (sinistra)
            keypoints[1] = bottom_row[-1]  # k2 (destra)
    
    elif 3 <= len(rows) < 5:  # Abbiamo alcune righe ma non tutte
        # Cerchiamo di mappare le righe che abbiamo alle righe del campo
        # Diciamo che abbiamo 3 righe, prenderemo top, middle e bottom
        
        # Righe disponibili in ordine dall'alto verso il basso
        available_rows = rows
        
        # Se abbiamo 3 righe: top, middle, bottom
        if len(available_rows) == 3:
            # Top row (k11, k12)
            top_row = available_rows[0]
            if len(top_row) >= 2:
                keypoints[10] = top_row[0]  # k11 (sinistra)
                keypoints[11] = top_row[-1]  # k12 (destra)
            
            # Middle row (potrebbe corrispondere a k8-k10 o k6-k7)
            # Scegliamo in base al numero di punti
            middle_row = available_rows[1]
            if len(middle_row) >= 3:  # Probabilmente k8-k10
                keypoints[7] = middle_row[0]  # k8
                keypoints[8] = middle_row[len(middle_row) // 2]  # k9
                keypoints[9] = middle_row[-1]  # k10
            elif len(middle_row) == 2:  # Potrebbe essere k6-k7
                keypoints[5] = middle_row[0]  # k6
                keypoints[6] = middle_row[-1]  # k7
            
            # Bottom row (k1, k2)
            bottom_row = available_rows[2]
            if len(bottom_row) >= 2:
                keypoints[0] = bottom_row[0]  # k1
                keypoints[1] = bottom_row[-1]  # k2
        
        # Se abbiamo 4 righe, possiamo dedurre quali sono in base al numero di punti
        elif len(available_rows) == 4:
            # Le righe con potenzialmente 3 punti sono la seconda dall'alto (k8-k10) e la quarta (k3-k5)
            row_candidates = [(i, len(row)) for i, row in enumerate(available_rows)]
            # Ordina per numero di punti decrescente
            row_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Identifica le righe che potrebbero avere 3 punti
            three_point_rows = [i for i, count in row_candidates if count >= 3]
            
            # Mappiamo in base alle posizioni identificate
            for i, row in enumerate(available_rows):
                if i == 0:  # Prima riga dall'alto (k11, k12)
                    if len(row) >= 2:
                        keypoints[10] = row[0]  # k11
                        keypoints[11] = row[-1]  # k12
                elif i == len(available_rows) - 1:  # Ultima riga (k1, k2)
                    if len(row) >= 2:
                        keypoints[0] = row[0]  # k1
                        keypoints[1] = row[-1]  # k2
                elif len(row) >= 3:  # Potenziale riga con 3 punti
                    if i == 1:  # Seconda riga (probabilmente k8-k10)
                        keypoints[7] = row[0]  # k8
                        keypoints[8] = row[len(row) // 2]  # k9
                        keypoints[9] = row[-1]  # k10
                    elif i == 2:  # Terza riga (potrebbe essere k3-k5)
                        keypoints[2] = row[0]  # k3
                        keypoints[3] = row[len(row) // 2]  # k4
                        keypoints[4] = row[-1]  # k5
                elif len(row) == 2:  # Riga con 2 punti (probabilmente k6-k7)
                    keypoints[5] = row[0]  # k6
                    keypoints[6] = row[-1]  # k7
    
    else:  # Abbiamo troppo poche righe
        print(f"Rilevate solo {len(rows)} righe di keypoints, è necessario stimare i keypoints mancanti.")
        # Proviamo a identificare almeno i keypoints agli angoli
        if rows:
            if len(rows[0]) >= 2:  # Prima riga
                keypoints[10] = rows[0][0]  # k11 (in alto a sinistra)
                keypoints[11] = rows[0][-1]  # k12 (in alto a destra)
            
            if len(rows) > 1 and len(rows[-1]) >= 2:  # Ultima riga
                keypoints[0] = rows[-1][0]  # k1 (in basso a sinistra)
                keypoints[1] = rows[-1][-1]  # k2 (in basso a destra)
    
    # Filtra solo i keypoints validi (non None)
    valid_keypoints = [kp for kp in keypoints if kp is not None]
    print(f"Rilevati {len(valid_keypoints)} keypoints validi su 12 necessari.")
    
    return valid_keypoints

def estimate_missing_keypoints(
    partial_keypoints: List[Tuple[float, float]], 
    image_shape: Tuple[int, int, int]
) -> List[Tuple[float, float]]:
    """
    Stima i keypoints mancanti basandosi sui keypoints rilevati e sulla geometria del campo.
    
    Args:
        partial_keypoints: Lista dei keypoints rilevati (può essere incompleta)
        image_shape: Dimensioni dell'immagine (h, w, c)
    
    Returns:
        Lista completa dei 12 keypoints stimati
    """
    # Se non abbiamo keypoints o ne abbiamo troppo pochi, ritorna quelli che abbiamo
    if len(partial_keypoints) < 4:
        print("Troppo pochi keypoints per stimare i mancanti.")
        return partial_keypoints
    
    h, w = image_shape[:2]
    
    # Crea array per i keypoints stimati
    keypoints = [(0, 0)] * 12
    
    # Mapping iniziale dei keypoints identificati
    # Organizza i keypoints disponibili in un array temporaneo
    # usando euristiche basate sulle posizioni relative
    if 4 <= len(partial_keypoints) < 12:
        # Ordina per y crescente (dall'alto al basso)
        sorted_by_y = sorted(partial_keypoints, key=lambda p: p[1])
        
        # Dividi in righe superiori e inferiori
        if len(sorted_by_y) >= 8:
            # Abbiamo abbastanza punti per dividere in 4 righe approssimative
            upper_points = sorted_by_y[:len(sorted_by_y)//2]
            lower_points = sorted_by_y[len(sorted_by_y)//2:]
            
            # Per ogni gruppo, dividi in 2 righe
            upper_top = upper_points[:len(upper_points)//2]
            upper_bottom = upper_points[len(upper_points)//2:]
            lower_top = lower_points[:len(lower_points)//2]
            lower_bottom = lower_points[len(lower_points)//2:]
            
            # Per ogni riga, ordina per x e assegna
            if upper_top:
                upper_top.sort(key=lambda p: p[0])
                if len(upper_top) >= 2:
                    keypoints[10] = upper_top[0]  # k11 (in alto a sinistra)
                    keypoints[11] = upper_top[-1]  # k12 (in alto a destra)
            
            if upper_bottom:
                upper_bottom.sort(key=lambda p: p[0])
                if len(upper_bottom) >= 3:
                    keypoints[7] = upper_bottom[0]  # k8
                    keypoints[8] = upper_bottom[len(upper_bottom)//2]  # k9
                    keypoints[9] = upper_bottom[-1]  # k10
                elif len(upper_bottom) == 2:
                    keypoints[7] = upper_bottom[0]  # k8
                    keypoints[9] = upper_bottom[-1]  # k10
            
            if lower_top:
                lower_top.sort(key=lambda p: p[0])
                if len(lower_top) >= 2:
                    keypoints[5] = lower_top[0]  # k6
                    keypoints[6] = lower_top[-1]  # k7
            
            if lower_bottom:
                lower_bottom.sort(key=lambda p: p[0])
                if len(lower_bottom) >= 3:
                    keypoints[2] = lower_bottom[0]  # k3
                    keypoints[3] = lower_bottom[len(lower_bottom)//2]  # k4
                    keypoints[4] = lower_bottom[-1]  # k5
                elif len(lower_bottom) == 2:
                    keypoints[2] = lower_bottom[0]  # k3
                    keypoints[4] = lower_bottom[-1]  # k5
        
        elif len(sorted_by_y) >= 4:
            # Abbiamo solo abbastanza punti per dividere in 2 righe
            upper_half = sorted_by_y[:len(sorted_by_y)//2]
            lower_half = sorted_by_y[len(sorted_by_y)//2:]
            
            upper_half.sort(key=lambda p: p[0])
            if len(upper_half) >= 2:
                keypoints[10] = upper_half[0]  # k11 (stima)
                keypoints[11] = upper_half[-1]  # k12 (stima)
            
            lower_half.sort(key=lambda p: p[0])
            if len(lower_half) >= 2:
                keypoints[0] = lower_half[0]  # k1 (stima)
                keypoints[1] = lower_half[-1]  # k2 (stima)
    
    # Filtra i keypoints identificati (non zero)
    identified = [(i, kp) for i, kp in enumerate(keypoints) if kp != (0, 0)]
    
    # Se non abbiamo identificato abbastanza keypoints, usa i keypoints parziali direttamente
    if len(identified) < 4:
        # Ordina i keypoints dall'alto a sinistra in basso a destra
        sorted_kps = sorted(partial_keypoints, key=lambda p: (p[1], p[0]))
        
        # Identifica gli angoli del campo (i quattro più esterni)
        if len(sorted_kps) >= 4:
            # Trova gli estremi sinistro, destro, alto, basso
            sorted_x = sorted(sorted_kps, key=lambda p: p[0])
            sorted_y = sorted(sorted_kps, key=lambda p: p[1])
            
            left_x = sorted_x[0][0]
            right_x = sorted_x[-1][0]
            top_y = sorted_y[0][1]
            bottom_y = sorted_y[-1][1]
            
            # Trova i punti più vicini agli angoli
            top_left = min(sorted_kps, key=lambda p: math.sqrt((p[0] - left_x)**2 + (p[1] - top_y)**2))
            top_right = min(sorted_kps, key=lambda p: math.sqrt((p[0] - right_x)**2 + (p[1] - top_y)**2))
            bottom_left = min(sorted_kps, key=lambda p: math.sqrt((p[0] - left_x)**2 + (p[1] - bottom_y)**2))
            bottom_right = min(sorted_kps, key=lambda p: math.sqrt((p[0] - right_x)**2 + (p[1] - bottom_y)**2))
            
            keypoints[10] = top_left      # k11
            keypoints[11] = top_right     # k12
            keypoints[0] = bottom_left    # k1
            keypoints[1] = bottom_right   # k2
    
    # Stima i keypoints rimanenti basandosi sui keypoints identificati
    # Caso 1: Abbiamo k1, k2, k11, k12 - possiamo stimare tutti gli altri
    if all(keypoints[i] != (0, 0) for i in [0, 1, 10, 11]):
        
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
        
        # Stima i punti centrali (k4, k9)
        if keypoints[3] == (0, 0) and keypoints[2] != (0, 0) and keypoints[4] != (0, 0):
            x3, y3 = keypoints[2]  # k3
            x5, y5 = keypoints[4]  # k5
            keypoints[3] = ((x3 + x5) / 2, (y3 + y5) / 2)  # k4 al centro tra k3 e k5
            
        if keypoints[8] == (0, 0) and keypoints[7] != (0, 0) and keypoints[9] != (0, 0):
            x8, y8 = keypoints[7]  # k8
            x10, y10 = keypoints[9]  # k10
            keypoints[8] = ((x8 + x10) / 2, (y8 + y10) / 2)  # k9 al centro tra k8 e k10
    
    # Applica correzioni aggiuntive per linee dritte
    keypoints = correct_keypoint_alignment(keypoints, image_shape)
    
    # Verifica la validità della geometria (se possibile)
    try:
        from utils.update_court_detection import verify_keypoint_structure
        if all(kp != (0, 0) for kp in keypoints):  # Se abbiamo tutti i keypoints
            if not verify_keypoint_structure(keypoints, image_shape):
                print("Attenzione: La struttura dei keypoints stimati potrebbe non essere valida")
    except (ImportError, Exception) as e:
        print(f"Impossibile verificare la struttura: {str(e)}")
    
    # Filtra solo i keypoints stimati (non (0, 0))
    estimated_keypoints = [kp for kp in keypoints if kp != (0, 0)]
    
    print(f"Stimati {len(estimated_keypoints)} keypoints su 12 necessari.")
    
    # Ordinamento finale dei keypoints stimati
    if len(estimated_keypoints) > 0:
        # Se non abbiamo esattamente 12 keypoints, ma ne abbiamo alcuni,
        # ordiniamoli per assicurarci che siano in un ordine coerente
        if len(estimated_keypoints) < 12:
            estimated_keypoints.sort(key=lambda p: (p[1], p[0]))  # Ordina prima per y, poi per x
    
    return estimated_keypoints

def correct_keypoint_alignment(
    keypoints: List[Tuple[float, float]],
    image_shape: Tuple[int, int, int]
) -> List[Tuple[float, float]]:
    """
    Corregge l'allineamento dei keypoints per garantire linee dritte.
    
    Args:
        keypoints: Lista dei keypoints da correggere
        image_shape: Dimensioni dell'immagine
        
    Returns:
        Lista di keypoints corretti
    """
    # Identifica i keypoints che appartengono alle linee verticali sinistra, centro e destra
    left_indices = [0, 2, 5, 7, 10]  # k1, k3, k6, k8, k11
    center_indices = [3, 8]  # k4, k9
    right_indices = [1, 4, 6, 9, 11]  # k2, k5, k7, k10, k12
    
    # Assicurati che i keypoints in ogni linea verticale abbiano la stessa coordinata x
    # Linea sinistra
    left_keypoints = [keypoints[i] for i in left_indices if keypoints[i] != (0, 0)]
    if len(left_keypoints) >= 2:
        avg_x = sum(kp[0] for kp in left_keypoints) / len(left_keypoints)
        for i in left_indices:
            if keypoints[i] != (0, 0):
                x, y = keypoints[i]
                keypoints[i] = (avg_x, y)
    
    # Linea destra
    right_keypoints = [keypoints[i] for i in right_indices if keypoints[i] != (0, 0)]
    if len(right_keypoints) >= 2:
        avg_x = sum(kp[0] for kp in right_keypoints) / len(right_keypoints)
        for i in right_indices:
            if keypoints[i] != (0, 0):
                x, y = keypoints[i]
                keypoints[i] = (avg_x, y)
    
    # Linea centrale
    center_keypoints = [keypoints[i] for i in center_indices if keypoints[i] != (0, 0)]
    if len(center_keypoints) >= 1:
        # Se abbiamo entrambi i keypoints laterali, possiamo calcolare il centro
        if keypoints[2] != (0, 0) and keypoints[4] != (0, 0):
            center_x = (keypoints[2][0] + keypoints[4][0]) / 2
            if keypoints[3] != (0, 0):
                keypoints[3] = (center_x, keypoints[3][1])
        
        if keypoints[7] != (0, 0) and keypoints[9] != (0, 0):
            center_x = (keypoints[7][0] + keypoints[9][0]) / 2
            if keypoints[8] != (0, 0):
                keypoints[8] = (center_x, keypoints[8][1])
    
    # Identifica i keypoints che appartengono alle linee orizzontali
    horizontal_rows = [
        [0, 1],      # k1, k2
        [2, 3, 4],   # k3, k4, k5
        [5, 6],      # k6, k7
        [7, 8, 9],   # k8, k9, k10
        [10, 11]     # k11, k12
    ]
    
    # Assicurati che i keypoints in ogni linea orizzontale abbiano la stessa coordinata y
    for row in horizontal_rows:
        row_keypoints = [keypoints[i] for i in row if keypoints[i] != (0, 0)]
        if len(row_keypoints) >= 2:
            avg_y = sum(kp[1] for kp in row_keypoints) / len(row_keypoints)
            for i in row:
                if keypoints[i] != (0, 0):
                    x, y = keypoints[i]
                    keypoints[i] = (x, avg_y)
    
    return keypoints

def improve_court_detection(
    image: np.ndarray,
    detected_keypoints: List[Tuple[float, float]],
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Migliora i keypoints rilevati combinando metodi di visione artificiale e
    conoscenza della geometria del campo da padel.
    
    Strategia:
    1. Se abbiamo già tutti i 12 keypoints, verificali e migliorali.
    2. Se ne mancano alcuni, usa i metodi CV per trovarli o stimarli.
    3. Se il rilevamento CV fallisce, stima i keypoints mancanti dalla geometria.
    
    Args:
        image: L'immagine del campo da padel
        detected_keypoints: I keypoints già rilevati (da YOLO o altro metodo)
        debug: Se True, mostra immagini intermedie per il debug
        
    Returns:
        Lista migliorata di tutti i keypoints
    """
    # Se abbiamo già 12 keypoints, verifichiamo che siano coerenti
    if len(detected_keypoints) == 12:
        print("Già rilevati 12 keypoints, verifica e ottimizzazione...")
        # Possiamo raffinare i keypoints esistenti usando tecniche CV
        # ad esempio cercando l'intersezione di linee più precisa
        return refine_keypoints(image, detected_keypoints, debug)
    
    # Se abbiamo alcuni keypoints ma non tutti, prova a trovare i mancanti
    if detected_keypoints:
        print(f"Rilevati {len(detected_keypoints)} keypoints, cerco i mancanti...")
        # Prima tenta con metodi CV
        cv_keypoints = detect_court_keypoints_cv(image, debug)
        
        # Se il rilevamento CV ha successo, combina i risultati
        if cv_keypoints:
            combined_keypoints = combine_keypoints(detected_keypoints, cv_keypoints, image.shape)
            if len(combined_keypoints) > len(detected_keypoints):
                print(f"Combinazione riuscita, ora abbiamo {len(combined_keypoints)} keypoints.")
                
                if len(combined_keypoints) == 12:
                    # Se ora abbiamo 12 keypoints, raffiniamo le posizioni
                    try:
                        from utils.update_court_detection import refine_keypoint_positions
                        # Prepara la maschera bianca per il raffinamento
                        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        _, _, v = cv2.split(hsv)
                        _, white_mask = cv2.threshold(v, 170, 255, cv2.THRESH_BINARY)
                        white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)
                        
                        refined_keypoints = refine_keypoint_positions(combined_keypoints, white_mask)
                        
                        # Correggi l'allineamento dei keypoints
                        from utils.update_court_detection import verify_keypoint_structure
                        
                        # Assicurati che sia coerente con la struttura attesa del campo
                        if verify_keypoint_structure(refined_keypoints, image.shape):
                            print("Struttura dei keypoints verificata e confermata.")
                            return refined_keypoints
                        else:
                            print("La struttura raffinata non è valida, utilizzo keypoints originali.")
                            return combined_keypoints
                    except (ImportError, Exception) as e:
                        print(f"Errore durante il raffinamento: {str(e)}")
                        return combined_keypoints
                
                return combined_keypoints
        
        # Se non miglioriamo, stima i keypoints mancanti
        print("Stima dei keypoints mancanti...")
        estimated = estimate_missing_keypoints(detected_keypoints, image.shape)
        
        # Verifica se abbiamo ora 12 keypoints
        if len(estimated) == 12:
            # Correggi l'allineamento dei keypoints per linee dritte
            estimated = correct_keypoint_alignment(estimated, image.shape)
        
        return estimated
    
    # Se non abbiamo keypoints, prova a trovarli con metodi CV
    print("Nessun keypoint rilevato, uso metodi CV per trovarli...")
    cv_keypoints = detect_court_keypoints_cv(image, debug)
    
    if not cv_keypoints:
        print("Impossibile rilevare keypoints con metodi CV.")
        return []
    
    # Se abbiamo keypoints ma meno di 12, proviamo a stimare i mancanti
    if len(cv_keypoints) < 12:
        cv_keypoints = estimate_missing_keypoints(cv_keypoints, image.shape)
    
    return cv_keypoints

def refine_keypoints(
    image: np.ndarray,
    keypoints: List[Tuple[float, float]],
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Raffina i keypoints esistenti cercando posizioni più precise basate
    sulle intersezioni delle linee del campo.
    
    Args:
        image: L'immagine del campo da padel
        keypoints: I keypoints da raffinare
        debug: Se True, mostra immagini intermedie per il debug
        
    Returns:
        Lista raffinata dei keypoints
    """
    # Implementazione del raffinamento
    # Questo potrebbe cercare intersezioni di linee vicino ai keypoints rilevati
    # per una localizzazione più precisa
    return keypoints  # Per ora, ritorna gli stessi keypoints

def combine_keypoints(
    yolo_keypoints: List[Tuple[float, float]],
    cv_keypoints: List[Tuple[float, float]],
    image_shape: Tuple[int, int, int]
) -> List[Tuple[float, float]]:
    """
    Combina keypoints rilevati da YOLO e metodi CV per ottenere
    il set più completo e preciso.
    
    Args:
        yolo_keypoints: Keypoints rilevati da YOLO
        cv_keypoints: Keypoints rilevati da metodi CV
        image_shape: Dimensioni dell'immagine
        
    Returns:
        Lista combinata dei keypoints
    """
    # Implementazione della combinazione
    # Potrebbe usare una fusione basata sulla confidenza o sulla distanza
    
    # Per semplicità, manteniamo i keypoints YOLO e aggiungiamo quelli CV mancanti
    h, w = image_shape[:2]
    
    # Distanza massima per considerare due keypoints come corrispondenti
    threshold = min(h, w) * 0.05
    
    # Copia i keypoints YOLO
    combined = list(yolo_keypoints)
    
    # Aggiungi i keypoints CV che non sono già presenti
    for cv_kp in cv_keypoints:
        # Verifica se questo keypoint CV è vicino a uno YOLO esistente
        if not any(euclidean_distance(cv_kp, yolo_kp) < threshold for yolo_kp in yolo_keypoints):
            combined.append(cv_kp)
    
    return combined

def auto_detect_court_keypoints(
    image: np.ndarray,
    yolo_model=None,
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Funzione principale per il rilevamento automatico dei keypoints del campo.
    Combina sia l'approccio basato su YOLO che quello basato su CV.
    
    Args:
        image: L'immagine del campo da padel
        yolo_model: Il modello YOLO per il rilevamento dei keypoints (opzionale)
        debug: Se True, mostra immagini intermedie per il debug
        
    Returns:
        Lista dei 12 keypoints del campo
    """
    keypoints = []
    
    # 1. Prova prima con YOLO se disponibile
    if yolo_model is not None:
        try:
            print("Tentativo di rilevamento keypoints con YOLO...")
            # Implementazione del rilevamento YOLO
            # ...
            
            # Per ora assumiamo che non abbiamo keypoints da YOLO
            keypoints = []
        except Exception as e:
            print(f"Errore nel rilevamento YOLO: {str(e)}")
            keypoints = []
    
    # 2. Se YOLO fallisce o non è disponibile, usa i metodi CV
    if not keypoints:
        print("Tentativo di rilevamento keypoints con metodi CV...")
        keypoints = detect_court_keypoints_cv(image, debug)
    
    # 3. Se abbiamo alcuni keypoints ma non tutti, prova a migliorarli
    if keypoints and len(keypoints) < 12:
        keypoints = improve_court_detection(image, keypoints, debug)
    
    return keypoints

def filter_and_reduce_lines(
    lines: List[np.ndarray], 
    white_mask: np.ndarray,
    max_lines: int = 100
) -> List[np.ndarray]:
    """
    Filtra le linee rilevate mantenendo solo quelle più significative.
    Implementa euristiche per ridurre il numero di linee e migliorare le prestazioni.
    
    Args:
        lines: Lista di linee rilevate dalla trasformata di Hough
        white_mask: Maschera binaria delle aree bianche
        max_lines: Numero massimo di linee da mantenere
        
    Returns:
        Lista di linee filtrate e ridotte
    """
    if not lines or len(lines) <= max_lines:
        return lines
    
    # Calcola la lunghezza di ogni linea e la percentuale di pixel bianchi
    line_info = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calcola punteggio in base alla lunghezza e alla percentuale di pixel bianchi
        num_samples = 20
        points_x = np.linspace(x1, x2, num_samples).astype(int)
        points_y = np.linspace(y1, y2, num_samples).astype(int)
        
        white_points = 0
        for x, y in zip(points_x, points_y):
            if 0 <= y < white_mask.shape[0] and 0 <= x < white_mask.shape[1]:
                if white_mask[y, x] > 0:
                    white_points += 1
        
        white_percentage = white_points / num_samples
        score = length * white_percentage  # Favorisce linee lunghe e completamente bianche
        
        # Calcola l'angolo della linea rispetto all'orizzontale
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        # Aggiungiamo un bonus per linee orizzontali e verticali
        if (angle < 10) or (angle > 170) or (80 < angle < 100):
            score *= 1.5  # Bonus per linee più orizzontali o verticali
        
        line_info.append((i, score, line))
    
    # Ordina le linee per punteggio decrescente
    line_info.sort(key=lambda x: x[1], reverse=True)
    
    # Assicuriamoci di avere un equilibrio tra linee orizzontali e verticali
    h_lines = []
    v_lines = []
    
    for _, _, line in line_info:
        x1, y1, x2, y2 = line
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        
        if (angle < 30) or (angle > 150):  # Linee orizzontali
            if len(h_lines) < max_lines // 2:
                h_lines.append(line)
        elif (angle > 60) and (angle < 120):  # Linee verticali
            if len(v_lines) < max_lines // 2:
                v_lines.append(line)
    
    # Combina le linee orizzontali e verticali
    filtered_lines = h_lines + v_lines
    
    # Se non abbiamo raggiunto il numero massimo, aggiungi altre linee in ordine di punteggio
    remaining = max_lines - len(filtered_lines)
    if remaining > 0:
        used_indices = set(h_lines + v_lines)
        for _, _, line in line_info:
            if line not in used_indices and remaining > 0:
                filtered_lines.append(line)
                remaining -= 1
    
    return filtered_lines

def orchestrate_court_detection(
    image: np.ndarray, 
    detection_method: str = "combined", 
    debug: bool = False
) -> List[Tuple[float, float]]:
    """
    Funzione di orchestrazione che gestisce il rilevamento dei keypoints della corte
    utilizzando diversi metodi in base alla configurazione.
    
    Args:
        image: L'immagine da analizzare
        detection_method: Metodo di rilevamento ("traditional", "advanced", o "combined") 
        debug: Se True, mostra immagini di debug
        
    Returns:
        Lista di keypoints rilevati
    """
    keypoints = []
    
    if detection_method in ["traditional", "combined"]:
        # Tentativo con metodo tradizionale
        print("Rilevamento corte: utilizzo metodo tradizionale basato sulle linee bianche...")
        keypoints_cv = detect_court_keypoints_cv(image, debug=debug)
        
        if len(keypoints_cv) == 12:
            print("Rilevati tutti i 12 keypoints con metodo tradizionale")
            
            # Mostra i keypoints per debug se richiesto
            if debug:
                debug_img = image.copy()
                for kp in keypoints_cv:
                    cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                cv2.imshow('Keypoints Tradizionali', debug_img)
                cv2.waitKey(500)
                cv2.imwrite('debug_cv_keypoints.jpg', debug_img)
            
            return keypoints_cv
        
        keypoints = keypoints_cv
        print(f"Rilevati {len(keypoints_cv)} keypoints con metodo tradizionale")
    
    if detection_method in ["advanced", "combined"]:
        # Tentativo con metodo avanzato
        from utils.advanced_court_detection import detect_court_keypoints_advanced
        print("Rilevamento corte: utilizzo metodo avanzato basato sulla maschera blu e sulle linee bianche...")
        keypoints_adv = detect_court_keypoints_advanced(image, debug=debug)
        
        # Mostra i keypoints avanzati per debug se richiesto
        if debug and keypoints_adv:
            debug_img = image.copy()
            for kp in keypoints_adv:
                cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
            cv2.imshow('Keypoints Avanzati', debug_img)
            cv2.waitKey(500)
        
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
                    for kp in combined_keypoints:
                        cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 5, (255, 255, 0), -1)
                    cv2.imshow('Keypoints Combinati', debug_img)
                    cv2.waitKey(500)
                    cv2.imwrite('debug_combined_keypoints.jpg', debug_img)
            elif len(keypoints_adv) > 0:
                combined_keypoints = keypoints_adv
            else:
                combined_keypoints = keypoints
            
            keypoints = combined_keypoints
    
    # Se abbiamo più di 12 keypoints dopo la combinazione, seleziona i migliori 12
    if len(keypoints) > 12:
        print(f"Rilevati {len(keypoints)} keypoints, selezione dei 12 migliori...")
        # Qui potremmo implementare una strategia di selezione più sofisticata
        # Per ora, prendiamo solo i primi 12
        keypoints = keypoints[:12]
    
    # Se abbiamo ancora keypoints incompleti, proviamo a stimarli
    if 0 < len(keypoints) < 12:
        print(f"Tentativo di miglioramento: stima dei keypoints mancanti ({len(keypoints)}/12)...")
        keypoints = improve_court_detection(image, keypoints, debug=debug)
        print(f"Dopo miglioramento: {len(keypoints)}/12 keypoints")
    
    return keypoints
