""" Functions to read and save videos """

from typing import Literal, Tuple
import cv2
import numpy as np
from pathlib import Path
import os

from utils import converters


def get_codec_for_video_format(file_path: str | Path) -> Tuple[int, str]:
    """
    Determina il codec appropriato per il formato del file video.
    
    Args:
        file_path: Percorso del file video
    
    Returns:
        Tuple[int, str]: Codec FourCC e nome del formato
    """
    path_str = str(file_path).lower()
    ext = os.path.splitext(path_str)[1]
    
    if ext == '.mp4':
        return cv2.VideoWriter_fourcc(*"mp4v"), "MP4"
    elif ext == '.mov':
        return cv2.VideoWriter_fourcc(*"XVID"), "MOV"  # XVID è compatibile con MOV
    elif ext == '.avi':
        return cv2.VideoWriter_fourcc(*"XVID"), "AVI"
    elif ext == '.mkv':
        return cv2.VideoWriter_fourcc(*"X264"), "MKV"
    elif ext in ['.wmv', '.asf']:
        return cv2.VideoWriter_fourcc(*"WMV2"), "WMV"
    else:
        # Formato predefinito
        return cv2.VideoWriter_fourcc(*"mp4v"), "MP4"


def read_video(
    path: str | Path, 
    max_frames: int = None,
) -> tuple[list[np.ndarray], int, int, int]:
    
    print("Reading Video ...")

    cap = cv2.VideoCapture(path)
    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame, 
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame_rgb)
        
        if max_frames is not None:
            if len(frames) >= max_frames:
                break

    cap.release()

    print("Done.")

    return frames, fps, w, h

def save_video(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int,
    h: int,
    w: int,
):
    # Usa la funzione di utilità per ottenere il codec appropriato
    fourcc, format_name = get_codec_for_video_format(path)
    print(f"Salvando il video in formato {format_name}...")
    
    out = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(
            frame, 
            cv2.COLOR_RGB2BGR,
        )
        out.write(frame_bgr)
    out.release()
    print(f"Video salvato con successo in {path}")