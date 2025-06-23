import cv2
import os
import tempfile
import numpy as np
from stitcher import (
    get_frames,
    cascade_frame_search,
    getValores,
    unir_cascade_frames_manteniendo_superior
)
import base64

def procesar_video_completo(
    video_path,
    crop_percent,
    keep_first_original,
    keep_last_original,
    interval,
    filtro,
    metodo_comparacion,
    umbral_coincidencia,
    incremento_paso,
    escala_reduccion,
    metodo_fusion
):
    frames = get_frames(video_path, crop_percent, keep_first_original, keep_last_original, interval)
    cascade_frames, cascade_keypoints, kp_inverse, direccion = cascade_frame_search(frames)
    cortes = getValores(
        cascade_frames, filtro, metodo_comparacion, umbral_coincidencia,
        incremento_paso, escala_reduccion, direccion
    )

    cascade_b64 = []
    for f in cascade_frames:
        _, buffer = cv2.imencode('.jpg', f)
        cascade_b64.append(base64.b64encode(buffer).decode('utf-8'))

    imagen_fusionada = unir_cascade_frames_manteniendo_superior(
        cascade_frames, cortes, metodo_fusion, direccion
    )

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, imagen_fusionada)

    return temp_file.name, cortes, cascade_b64, direccion, metodo_fusion

def regenerar_imagen_nueva(cascade_frames, cortes, metodo_fusion, direccion):
    imagen_fusionada = unir_cascade_frames_manteniendo_superior(
        cascade_frames, cortes, metodo_fusion, direccion
    )
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, imagen_fusionada)
    return temp_file.name
