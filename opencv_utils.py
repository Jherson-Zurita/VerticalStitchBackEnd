import cv2
import os
import tempfile
import numpy as np
from stitcher import (
    get_frames, cascade_frame_search, getValores,
    unir_cascade_frames_manteniendo_superior
)

# Guarda estado para regenerar imagen
_last_cascade_frames = None
_last_direccion = None
_last_metodo_fusion = None

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
    global _last_cascade_frames, _last_direccion, _last_metodo_fusion

    frames = get_frames(video_path, crop_percent, keep_first_original, keep_last_original, interval)
    cascade_frames, cascade_keypoints, kp_inverse, direccion = cascade_frame_search(frames)
    cortes = getValores(
        cascade_frames, filtro, metodo_comparacion, umbral_coincidencia,
        incremento_paso, escala_reduccion, direccion
    )

    _last_cascade_frames = cascade_frames
    _last_direccion = direccion
    _last_metodo_fusion = metodo_fusion

    imagen_fusionada = unir_cascade_frames_manteniendo_superior(
        cascade_frames, cortes, metodo_fusion, direccion
    )
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, imagen_fusionada)
    return temp_file.name, cortes

def regenerar_imagen_nueva(cortes):
    global _last_cascade_frames, _last_direccion, _last_metodo_fusion

    print("== Debug Info ==")
    print("Tipo de _last_cascade_frames:", type(_last_cascade_frames))
    print("Tamaño de _last_cascade_frames:", len(_last_cascade_frames) if _last_cascade_frames else "None")
    print("_last_direccion:", _last_direccion)
    print("_last_metodo_fusion:", _last_metodo_fusion)
    print("Tipo de cortes:", type(cortes))
    print("Valor de cortes:", cortes)
    print("=================")

    if _last_cascade_frames is None:
        raise ValueError("No hay datos previos de procesamiento")
    
    imagen_fusionada = unir_cascade_frames_manteniendo_superior(
        _last_cascade_frames, cortes, _last_metodo_fusion, _last_direccion
    )

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, imagen_fusionada)
    return temp_file.name

