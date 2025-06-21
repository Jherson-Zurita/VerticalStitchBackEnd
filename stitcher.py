import cv2
import numpy as np
def get_frames(video_path, crop_percent=0, keep_first_original=True, keep_last_original=True, interval=1):
    # Abre el video con OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Lee los frames del video
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Añade frames según el intervalo especificado
        if i % interval == 0 or i == 0 or i == frame_count - 1:  # Siempre incluye el primer y último frame
            # Aplica recorte al frame si no es el primer o último frame o si no se deben conservar originales
            if (i != 0 or not keep_first_original) and (i != frame_count - 1 or not keep_last_original):
                height = frame.shape[0]
                crop_height = int(height * (1 - crop_percent / 100))
                frame = frame[:crop_height, :]

            # Añade cada frame a la lista
            frames.append(frame)

    # Libera el video
    cap.release()
    return frames

def match_keypoints(desc1, desc2):
    """
    Encuentra coincidencias entre descriptores.
    """
    # Validar que los descriptores no sean None y tengan dimensiones válidas
    if desc1 is None or desc2 is None:
        return []  # No hay coincidencias si alguno es None
    
    if desc1.shape[1] != desc2.shape[1]:
        return []  # No hay coincidencias si las dimensiones no coinciden
    
    # Crear matcher de fuerza bruta
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Encontrar coincidencias
    matches = bf.match(desc1, desc2)
    
    # Ordenar coincidencias por distancia
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

def detect_keypoints_in_areas(frame, area="both"):
    """
    Detecta puntos clave en la mitad superior, inferior o ambas del frame.

    :param frame: Imagen de entrada
    :param area: Área donde buscar puntos clave ("upper", "lower" o "both")
    :return: Puntos clave y descriptores de la(s) área(s) seleccionada(s)
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Crear detector ORB
    orb = cv2.ORB_create(nfeatures=1000)

    # Dividir el frame en dos mitades
    h, w = gray.shape
    upper_half = gray[:h//2, :]
    lower_half = gray[h//2:, :]

    # Inicializar listas para puntos clave y descriptores
    keypoints = []
    descriptors = []

    # Detectar puntos clave según el área especificada
    if area == "upper" or area == "both":
        upper_keypoints, upper_descriptors = orb.detectAndCompute(upper_half, None)
        # Ajustar coordenadas de puntos clave
        adjusted_upper_keypoints = [
            cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size) for kp in upper_keypoints
        ]
        keypoints.extend(adjusted_upper_keypoints)
        if upper_descriptors is not None:
            descriptors.append(upper_descriptors)

    if area == "lower" or area == "both":
        lower_keypoints, lower_descriptors = orb.detectAndCompute(lower_half, None)
        # Ajustar coordenadas de puntos clave
        adjusted_lower_keypoints = [
            cv2.KeyPoint(kp.pt[0], kp.pt[1] + h//2, kp.size) for kp in lower_keypoints
        ]
        keypoints.extend(adjusted_lower_keypoints)
        if lower_descriptors is not None:
            descriptors.append(lower_descriptors)

    # Combinar descriptores si se detectaron en ambas áreas
    if descriptors:
        descriptors = cv2.vconcat(descriptors)
    else:
        descriptors = None

    return keypoints, descriptors

def cascade_frame_search(frames):
    frame_count = len(frames)
    cascade_frames = []
    cascade_keypoints = []
    last_position = 0
    
    # Detect initial keypoints in the first and last frames
    first_frame = frames[0]
    last_frame = frames[-1]
    first_upper_kp, first_upper_desc = detect_keypoints_in_areas(first_frame, area="upper")
    last_lower_kp, last_lower_desc = detect_keypoints_in_areas(last_frame, area="lower")
    
    # Start cascade from first frame if it has fewer keypoints
    if len(first_upper_kp) < len(last_lower_kp):
        current_desc = first_upper_desc
        current_kp = first_upper_kp
        cascade_frames.append(first_frame)
        cascade_keypoints.append(first_upper_kp)
        search_direction = 'forward'
    else:
        current_desc = last_lower_desc
        current_kp = last_lower_kp
        cascade_frames.append(last_frame)
        cascade_keypoints.append(last_lower_kp)
        search_direction = 'backward'
        last_position = frame_count - 1
    
    while True:
        best_match_percentage = 0
        best_frame = None
        best_frame_kp = None
        best_frame_desc = None
        best_frame_index = 0

        # Define search range based on direction
        if search_direction == 'forward':
            search_range = range(last_position + 1, frame_count)
        else:
            search_range = range(last_position - 1, -1, -1)

        for i in search_range:
            frame = frames[i]

            # Detect keypoints in search area
            if search_direction == 'forward':
                frame_kp, frame_desc = detect_keypoints_in_areas(frame, area="lower")
            else:
                frame_kp, frame_desc = detect_keypoints_in_areas(frame, area="upper")

            # Match keypoints
            matches = match_keypoints(current_desc, frame_desc)
            match_percentage = len(matches) / len(current_kp) * 100 if current_kp else 0

            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage
                best_frame = frame
                best_frame_kp = frame_kp
                best_frame_desc = frame_desc
                best_frame_index = i

        # Stop if no good match found
        if best_match_percentage < 5 or best_frame is None:
            break

        # Add best frame to cascade
        cascade_frames.append(best_frame)
        cascade_keypoints.append(best_frame_kp)

        # Update for next iteration
        if best_match_percentage >= 5:
            if search_direction == 'forward':
                current_kp, current_desc = detect_keypoints_in_areas(best_frame, area="upper")
            else:
                current_kp, current_desc = detect_keypoints_in_areas(best_frame, area="lower")
        last_position = best_frame_index

    # Comparar similitudes y eliminar frames similares
    to_remove = set()
    for i in range(len(cascade_frames) - 1):
        similarity = compare_features(cascade_frames[i], cascade_frames[i + 1])
        if similarity > 0.80:
            to_remove.update(range(i + 1, len(cascade_frames)))  # Marcar frames a eliminar

    # Filtrar frames y keypoints
    cascade_frames = [frame for i, frame in enumerate(cascade_frames) if i not in to_remove]
    cascade_keypoints = [kp for i, kp in enumerate(cascade_keypoints) if i not in to_remove]


    # Detect keypoints in the inverse area for the first/last frame
    if search_direction == 'forward':
        last_frame_inverse_kp, _ = detect_keypoints_in_areas(last_frame, area="lower")
        return cascade_frames, cascade_keypoints, last_frame_inverse_kp, search_direction
    else:
        first_frame_inverse_kp, _ = detect_keypoints_in_areas(first_frame, area="upper")
        return cascade_frames, cascade_keypoints, first_frame_inverse_kp, search_direction

def find_best_matching_frame(frames):
    """
    Busca el frame con mayor porcentaje de coincidencias y retorna todo lo necesario para visualización.
    """
    frame_count = len(frames)
    
    # Inicializar variables para guardar el mejor frame encontrado
    best_match_percentage = 0
    best_frame_index = 0
    best_frame = None
    best_matches_upper = None
    best_matches_lower = None
    best_intermediate_upper_kp = None
    best_intermediate_lower_kp = None
    
    # Detectar puntos clave en el primer y último frame
    first_frame = frames[0]
    last_frame = frames[-1]
    first_upper_kp, first_upper_desc = detect_keypoints_in_areas(first_frame, area="upper")
    last_lower_kp, last_lower_desc = detect_keypoints_in_areas(last_frame, area="lower")
    
    current_upper_desc = first_upper_desc
    current_upper_kp = first_upper_kp
    current_lower_desc = last_lower_desc
    current_lower_kp = last_lower_kp
    
    for i in range(1, frame_count - 1):
        intermediate_frame = frames[i]
        
        # Detectar puntos clave en el frame intermedio
        inter_upper_kp, inter_upper_desc = detect_keypoints_in_areas(intermediate_frame, area="upper")
        inter_lower_kp, inter_lower_desc = detect_keypoints_in_areas(intermediate_frame, area="lower")
        
        # Encontrar coincidencias CRUZADAS
        matches_upper = match_keypoints(current_upper_desc, inter_lower_desc)
        matches_lower = match_keypoints(current_lower_desc, inter_upper_desc)
        
        # Calcular porcentaje de coincidencias
        match_percentage_upper = len(matches_upper) / len(current_upper_kp) * 100 if current_upper_kp else 0
        match_percentage_lower = len(matches_lower) / len(current_lower_kp) * 100 if current_lower_kp else 0
        
        # Combinar porcentajes de coincidencias
        total_match_percentage = (match_percentage_upper + match_percentage_lower) / 2
        
        # Actualizar el mejor frame si es necesario
        if total_match_percentage > best_match_percentage:
            best_match_percentage = total_match_percentage
            best_frame_index = i
            best_frame = intermediate_frame
            best_matches_upper = matches_upper
            best_matches_lower = matches_lower
            best_intermediate_upper_kp = inter_upper_kp
            best_intermediate_lower_kp = inter_lower_kp
               
    # Si no se encontró un buen frame intermedio
    if best_frame is None:
        return None,None,None, current_upper_kp,[], [], current_lower_kp,[], []
    
    if best_match_percentage < 50:
        return None,None,None, current_upper_kp,[], [], current_lower_kp,[], []
    else:
        if (match_percentage_lower < 20):
            return None,None,None, current_upper_kp,[], [], current_lower_kp,[], []
        elif (match_percentage_upper < 20):
            return None,None,None, current_upper_kp,[], [], current_lower_kp,[], []
        else:
            # Retornar todo lo necesario para visualización
            return (
                first_frame, last_frame, best_frame,
                first_upper_kp, best_intermediate_lower_kp, best_matches_upper,
                last_lower_kp, best_intermediate_upper_kp, best_matches_lower
            )

def compare_features(imageA, imageB):
    # Inicializar el detector ORB
    orb = cv2.ORB_create()

    # Encontrar los puntos clave y descriptores
    kpA, desA = orb.detectAndCompute(imageA, None)
    kpB, desB = orb.detectAndCompute(imageB, None)

    # Usar un matcher para encontrar coincidencias
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desA, desB)

    # Calcular la similitud
    similarity = len(matches) / max(len(kpA), len(kpB))
    return similarity

def aplicar_filtro(imagen, filtro='canny', umbral_bajo=10, umbral_alto=200):
    """
    Aplica diversos filtros a una imagen para mejorar la detección de bordes.
    
    Args:
        imagen: Imagen de entrada en formato BGR
        filtro: Tipo de filtro ('canny', 'sobel', 'laplacian', 'scharr')
        umbral_bajo: Umbral bajo para el detector Canny
        umbral_alto: Umbral alto para el detector Canny
        
    Returns:
        Imagen filtrada
    """
    if filtro == 'normal':
        # Retorna la imagen original sin modificaciones
        return imagen
    
    # Lee la imagen en escala de grises
    img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Reducción de ruido
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Aplica el filtro seleccionado
    if filtro == 'canny':
        img_filtro = cv2.Canny(img_blurred, umbral_bajo, umbral_alto)
    elif filtro == 'sobel':
        sobel_x = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)
        img_filtro = cv2.convertScaleAbs(cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0))
    elif filtro == 'laplacian':
        img_filtro = cv2.convertScaleAbs(cv2.Laplacian(img_blurred, cv2.CV_64F))
    elif filtro == 'scharr':
        scharr_x = cv2.Scharr(img_blurred, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(img_blurred, cv2.CV_64F, 0, 1)
        img_filtro = cv2.convertScaleAbs(cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0))
    else:
        raise ValueError("Filtro no válido. Use 'canny', 'sobel', 'laplacian' o 'scharr'.")
    
    return img_filtro

def calcular_coincidencia(img1, img2, metodo='pixel'):
    """
    Calcula la coincidencia entre dos imágenes utilizando diferentes métodos.
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        metodo: Método de comparación ('pixel', 'ssim', 'histograma', 'orb')
        
    Returns:
        Puntuación de coincidencia entre 0 y 1
    """
    if img1.shape != img2.shape:
        return 0.0
    
    if metodo == 'pixel':
        # Comparación pixel a pixel
        return np.sum(img1 == img2) / img1.size
    elif metodo == 'ssim':
        # Índice de similitud estructural
        return np.sum(img1 == img2) / img1.size
    elif metodo == 'histograma':
        # Comparación por histograma
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    elif metodo == 'orb':
        # Utilizando características ORB
        try:
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return 0.0
                
            # Crear BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # Normalizar puntuación
            return len(matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0.0
        except:
            print("Error en la detección ORB, usando comparación pixel a pixel")
            return np.sum(img1 == img2) / img1.size
    else:
        raise ValueError("Método no válido. Use 'pixel', 'ssim', 'histograma' o 'orb'.")

def comparar_imagenes(img1, img2, filtro='canny', metodo_comparacion='combinado', umbral_coincidencia=0.80, 
                      incremento_paso=1, escala_reduccion=0.9):
    """
    Compara dos imágenes buscando una parte de la primera en la segunda.
    
    Args:
        img1: Primera imagen (imagen de referencia)
        img2: Segunda imagen (imagen donde buscar)
        filtro: Tipo de filtro a aplicar ('canny', 'sobel', 'laplacian', 'scharr')
        metodo_comparacion: Método de comparación ('pixel', 'ssim', 'histograma', 'orb', 'combinado')
        umbral_coincidencia: Umbral para considerar una coincidencia buena (0-1)
        incremento_paso: Incremento de posición en cada iteración
        escala_reduccion: Factor de reducción para la imagen de referencia

        
    Returns:
        Diccionario con la mejor posición, porcentaje de coincidencia y más información
    """
    # Aplicar filtro a la primera imagen y dividir en partes superior e inferior
    img1_canny = aplicar_filtro(img1, filtro=filtro)
    height, width = img1_canny.shape
    img1_superior = img1_canny[:height//2, :]  # Mitad superior de img1_canny
    referencia_height, referencia_width = img1_superior.shape

    # Aplicar filtro a la segunda imagen
    img2_canny = aplicar_filtro(img2, filtro=filtro)

    # Inicializar variables para seguimiento
    mejor_coincidencia = 0
    mejor_posicion = None
    mejor_altura_referencia = referencia_height
    y_inicial = referencia_height  # Empezar desde la altura de la referencia
    resultados_intermedios = []

    # Variables para diferentes métodos de comparación
    metodos = ['pixel', 'ssim', 'histograma', 'orb'] if metodo_comparacion == 'combinado' else [metodo_comparacion]
    pesos = {'pixel': 0.4, 'ssim': 0.3, 'histograma': 0.2, 'orb': 0.1}  # Pesos para el método combinado
    
    # Primera búsqueda: escaneo vertical desde arriba
    y = y_inicial
    while referencia_height >= img1_canny.shape[0] // 4:
        # Reiniciar posición y para cada altura de referencia diferente
        y = referencia_height
        while y <= img2_canny.shape[0]:
            # Verificar que tenemos suficiente imagen para comparar
            if y < referencia_height or y > img2_canny.shape[0]:
                y += incremento_paso
                continue
                
            # Recorte de img2_canny del mismo tamaño que la referencia
            try:
                recorte_img2 = img2_canny[y - referencia_height:y, :]
                
                # Verificar dimensiones
                if recorte_img2.shape != img1_superior.shape:
                    y += incremento_paso
                    continue
                    
                # Calcular coincidencia según el método seleccionado
                coincidencia = 0
                if metodo_comparacion == 'combinado':
                    for metodo in metodos:
                        try:
                            coincidencia_metodo = calcular_coincidencia(img1_superior, recorte_img2, metodo)
                            coincidencia += coincidencia_metodo * pesos[metodo]
                        except:
                            # Si un método falla, continuamos con los otros
                            pass
                else:
                    coincidencia = calcular_coincidencia(img1_superior, recorte_img2, metodo_comparacion)
                
                # Almacenar resultados intermedios
                resultados_intermedios.append((y, referencia_height, coincidencia))
                
                # Actualizar mejor coincidencia
                if coincidencia > mejor_coincidencia:
                    mejor_coincidencia = coincidencia
                    mejor_posicion = y
                    mejor_altura_referencia = referencia_height
                
                # Si la coincidencia es muy buena, podemos terminar
                if coincidencia >= umbral_coincidencia:
                    break
                    
            except Exception as e:
                print(f"Error en y={y}: {e}")
            
            y += incremento_paso
        
        # Si ya encontramos una coincidencia suficientemente buena, terminar
        if mejor_coincidencia >= umbral_coincidencia:
            break
            
        # Reducir altura de referencia para siguiente iteración
        referencia_height = int(referencia_height * escala_reduccion)
        if referencia_height < 10:  # Evitar alturas demasiado pequeñas
            break
        img1_superior = img1_canny[:referencia_height, :]

    # Segunda búsqueda: buscar en áreas cercanas a la mejor coincidencia para refinar
    if mejor_posicion is not None:
        # Definir área de búsqueda refinada
        rango_y = 20  # Buscar 20 píxeles arriba y abajo
        paso_refinado = max(1, incremento_paso // 2)
        
        # Usar la altura de referencia que dio la mejor coincidencia
        img1_superior = img1_canny[:mejor_altura_referencia, :]
        
        y_start = max(mejor_altura_referencia, mejor_posicion - rango_y)
        y_end = min(img2_canny.shape[0], mejor_posicion + rango_y)
        
        for y in range(y_start, y_end, paso_refinado):
            try:
                recorte_img2 = img2_canny[y - mejor_altura_referencia:y, :]
                
                if recorte_img2.shape != img1_superior.shape:
                    continue
                
                # Usar el método que dio mejores resultados en la primera búsqueda
                if metodo_comparacion == 'combinado':
                    coincidencia = 0
                    for metodo in metodos:
                        try:
                            coincidencia_metodo = calcular_coincidencia(img1_superior, recorte_img2, metodo)
                            coincidencia += coincidencia_metodo * pesos[metodo]
                        except:
                            pass
                else:
                    coincidencia = calcular_coincidencia(img1_superior, recorte_img2, metodo_comparacion)
                
                if coincidencia > mejor_coincidencia:
                    mejor_coincidencia = coincidencia
                    mejor_posicion = y
            except:
                pass

    # Preparar resultados
    resultados = {
        "posicion_y": mejor_posicion,
        "altura_referencia": mejor_altura_referencia,
        "coincidencia": mejor_coincidencia,
        "filtro_usado": filtro,
        "metodo_comparacion": metodo_comparacion,
        "umbral_alcanzado": mejor_coincidencia >= umbral_coincidencia
    }
    
    return resultados

def unir_cascade_frames_manteniendo_superior(cascade_frames, cortes, metodo_fusion='superposicion', direccion='forwards'):
    """
    Une imágenes en cascade_frames manteniendo secciones desde la parte superior,
    cortando el resto, y apilándolas sobre la primera imagen. Incluye métodos para
    evitar líneas negras en las uniones.
    
    :param cascade_frames: Lista de arreglos NumPy resultantes de la búsqueda en cascada.
    :param cortes: Lista de alturas a mantener desde la parte superior.
    :param metodo_fusion: Método para fusionar las imágenes ('simple', 'superposicion', 'gradiente', 'costura')
    :return: Imagen resultante como arreglo NumPy.
    """ 
    if len(cascade_frames) - 1 != len(cortes):
        raise ValueError("El número de cortes debe ser igual al número de transiciones entre imágenes.")

    if direccion == 'backward':
        cascade_frames = cascade_frames[::-1]

    # Asegurarse de que todas las imágenes tengan el mismo ancho
    ancho_base = cascade_frames[0].shape[1]

    # Método 1: Unión simple mejorada
    if metodo_fusion == 'simple':
        resultado = cascade_frames[0].copy()  # Usar copy() para evitar modificar el original
        for i in range(len(cortes)):
            altura_mantener = cortes[i]
            seccion_mantener = cascade_frames[i + 1][:altura_mantener, :, :].copy()

            nueva_altura = resultado.shape[0] + seccion_mantener.shape[0]
            # Usar el mismo tipo de dato que la imagen para evitar conversiones
            lienzo = np.zeros((nueva_altura, ancho_base, resultado.shape[2]), dtype=resultado.dtype)

            # Colocar las imágenes con precisión
            lienzo[:seccion_mantener.shape[0], :, :] = seccion_mantener
            lienzo[seccion_mantener.shape[0]:, :, :] = resultado

            resultado = lienzo

    # Método 2: Superposición con zona de mezcla
    elif metodo_fusion == 'superposicion':
        resultado = cascade_frames[0].copy()
        for i in range(len(cortes)):
            altura_mantener = cortes[i]
            
            # Crear una zona de solapamiento de algunos píxeles
            zona_solapamiento = 3  # píxeles de solapamiento
            
            # Asegurarse de que no exceda los límites
            if altura_mantener - zona_solapamiento < 0:
                zona_solapamiento = 0
                
            # Sección a mantener incluyendo la zona de solapamiento
            if zona_solapamiento > 0:
                seccion_mantener = cascade_frames[i + 1][:altura_mantener, :, :].copy()
                
                # Preparar lienzo con el tamaño adecuado
                nueva_altura = resultado.shape[0] + altura_mantener - zona_solapamiento
                lienzo = np.zeros((nueva_altura, ancho_base, resultado.shape[2]), dtype=resultado.dtype)
                
                # Colocar la sección superior
                lienzo[:altura_mantener, :, :] = seccion_mantener
                
                # Mezclar gradualmente en la zona de solapamiento
                for j in range(zona_solapamiento):
                    alpha = j / zona_solapamiento
                    beta = 1.0 - alpha
                    
                    fila_destino = altura_mantener - zona_solapamiento + j
                    fila_origen = j
                    
                    lienzo[fila_destino, :, :] = (beta * seccion_mantener[fila_destino, :, :] + 
                                                 alpha * resultado[fila_origen, :, :]).astype(resultado.dtype)
                
                # Colocar el resto de la imagen resultado
                lienzo[altura_mantener:, :, :] = resultado[zona_solapamiento:, :, :]
            else:
                # Sin solapamiento, usar el método simple
                seccion_mantener = cascade_frames[i + 1][:altura_mantener, :, :].copy()
                nueva_altura = resultado.shape[0] + seccion_mantener.shape[0]
                lienzo = np.zeros((nueva_altura, ancho_base, resultado.shape[2]), dtype=resultado.dtype)
                lienzo[:seccion_mantener.shape[0], :, :] = seccion_mantener
                lienzo[seccion_mantener.shape[0]:, :, :] = resultado
            
            resultado = lienzo

    # Método 3: Fusión con degradado (gradient blending)
    elif metodo_fusion == 'gradiente':
        resultado = cascade_frames[0].copy()
        for i in range(len(cortes)):
            altura_mantener = cortes[i]
        
            # Definir una zona de transición para el degradado
            zona_transicion = 20  # píxeles para el degradado
        
            # Ajustar si la zona de transición es mayor que la altura a mantener
            if altura_mantener <= zona_transicion:
                zona_transicion = max(1, altura_mantener // 2)
        
            seccion_mantener = cascade_frames[i + 1][:altura_mantener, :, :].copy()
        
            # Crear máscara de degradado
            mascara = np.ones((altura_mantener, ancho_base), dtype=np.float32)
            for j in range(zona_transicion):
                # Crear un degradado suave en la parte inferior de la sección
                mascara[altura_mantener - zona_transicion + j, :] = j / zona_transicion
        
            # Expandir dimensiones de la máscara para aplicarla a los canales de color
            mascara = np.expand_dims(mascara, axis=2)
            mascara = np.repeat(mascara, resultado.shape[2], axis=2)
        
            # Aplicar la máscara a la sección a mantener
            seccion_con_degradado = seccion_mantener.astype(np.float32) * mascara
        
            # Preparar lienzo y combinar imágenes
            nueva_altura = resultado.shape[0] + altura_mantener - zona_transicion
            lienzo = np.zeros((nueva_altura, ancho_base, resultado.shape[2]), dtype=np.float32)
        
            # Colocar sección superior con degradado
            lienzo[:altura_mantener, :, :] = seccion_con_degradado
        
            # Crear máscara inversa para la parte de abajo
            mascara_inversa = np.ones((zona_transicion, ancho_base, resultado.shape[2]), dtype=np.float32)
            for j in range(zona_transicion):
                mascara_inversa[j, :, :] = 1 - (j / zona_transicion)
        
            # Aplicar degradado a la parte superior de la imagen resultado
            parte_superior_resultado = resultado[:zona_transicion, :, :].astype(np.float32) * mascara_inversa

            # Combinar la parte que se solapa
            lienzo[altura_mantener - zona_transicion:altura_mantener, :, :] += parte_superior_resultado
        
            # Añadir el resto de la imagen resultado
            lienzo[altura_mantener:, :, :] = resultado[zona_transicion:, :, :].astype(np.float32)
        
            # Convertir el resultado final a uint8
            resultado = np.clip(lienzo, 0, 255).astype(np.uint8)
    
    # Método 4: Costura de imágenes con Seam Carving
    elif metodo_fusion == 'costura':
        resultado = cascade_frames[0].copy()
        for i in range(len(cortes)):
            altura_mantener = cortes[i]
            altura_mantener = int(altura_mantener)
            seccion_mantener = cascade_frames[i + 1][:altura_mantener, :, :].copy()
            
            # Zona de solapamiento para encontrar la mejor costura
            zona_solapamiento = 3
            
            # Ajustar si la zona es mayor que las imágenes
            if altura_mantener < zona_solapamiento or resultado.shape[0] < zona_solapamiento:
                zona_solapamiento = min(altura_mantener, resultado.shape[0]) // 2
                
            if zona_solapamiento > 0:
                # Extraer regiones de solapamiento
                region_superior = seccion_mantener[altura_mantener - zona_solapamiento:, :, :]
                region_inferior = resultado[:zona_solapamiento, :, :]
                
                # Calcular la diferencia absoluta entre las regiones
                diferencia = np.sum(np.abs(region_superior.astype(np.float32) - 
                                          region_inferior.astype(np.float32)), axis=2)
                
                # Encontrar la costura óptima usando programación dinámica
                costura = np.zeros((zona_solapamiento, ancho_base), dtype=np.int32)
                energia = diferencia.copy()
                
                # Calcular energía acumulada
                for j in range(1, zona_solapamiento):
                    for k in range(ancho_base):
                        # Encontrar el camino de menor energía
                        if k == 0:
                            idx_min = np.argmin(energia[j-1, :2])
                        elif k == ancho_base - 1:
                            idx_min = np.argmin(energia[j-1, k-1:]) + k - 1
                        else:
                            idx_min = np.argmin(energia[j-1, k-1:k+2]) + k - 1
                            
                        costura[j, k] = idx_min
                        energia[j, k] += energia[j-1, idx_min]
                
                # Crear máscara basada en la costura
                mascara = np.zeros((zona_solapamiento, ancho_base), dtype=np.float32)
                
                # Inicializar en la parte inferior basado en la energía mínima
                columna_actual = np.argmin(energia[-1, :])
                mascara[-1, :columna_actual+1] = 1
                
                # Seguir la costura hacia arriba
                for j in range(zona_solapamiento-2, -1, -1):
                    columna_actual = costura[j+1, columna_actual]
                    mascara[j, :columna_actual+1] = 1
                
                # Suavizar la máscara para evitar transiciones bruscas
                mascara = cv2.GaussianBlur(mascara, (5, 5), 0)
                
                # Expandir dimensiones para aplicar a canales de color
                mascara = np.expand_dims(mascara, axis=2)
                mascara = np.repeat(mascara, resultado.shape[2], axis=2)
                
                # Preparar lienzo con tamaño adecuado
                nueva_altura = resultado.shape[0] + altura_mantener - zona_solapamiento
                lienzo = np.zeros((nueva_altura, ancho_base, resultado.shape[2]), dtype=resultado.dtype)
                
                # Colocar sección superior completa
                lienzo[:altura_mantener - zona_solapamiento, :, :] = seccion_mantener[:altura_mantener - zona_solapamiento, :, :]
                
                # Colocar zona de solapamiento mezclada
                region_mezclada = (region_superior * mascara + region_inferior * (1 - mascara)).astype(resultado.dtype)
                lienzo[altura_mantener - zona_solapamiento:altura_mantener, :, :] = region_mezclada
                
                # Colocar resto de la imagen resultado
                lienzo[altura_mantener:, :, :] = resultado[zona_solapamiento:, :, :]
            else:
                # Sin solapamiento, usar método simple
                nueva_altura = resultado.shape[0] + altura_mantener
                lienzo = np.zeros((nueva_altura, ancho_base, resultado.shape[2]), dtype=resultado.dtype)
                lienzo[:altura_mantener, :, :] = seccion_mantener
                lienzo[altura_mantener:, :, :] = resultado
            
            resultado = lienzo
    
    else:
        raise ValueError("Método de fusión no reconocido. Use 'simple', 'superposicion', 'gradiente' o 'costura'.")
    
    # Asegurar que los valores estén dentro del rango válido
    if resultado.dtype == np.uint8:
        resultado = np.clip(resultado, 0, 255)
    
    return resultado

def getValores(cascade_frames,filtro_, metodo_comparacion_, umbral_coincidencia_, incremento_paso_, escala_reduccion_, direccion='forward'):
    cortes = []
    
    if direccion == 'backward':
        cascade_frames = cascade_frames[::-1]
    
    for frame in range(len(cascade_frames)-1):
        resultado = comparar_imagenes(cascade_frames[frame], cascade_frames[frame + 1], filtro=filtro_, metodo_comparacion=metodo_comparacion_, umbral_coincidencia=umbral_coincidencia_, incremento_paso=incremento_paso_, escala_reduccion=escala_reduccion_)
        cortes.append((resultado['posicion_y'] - resultado['altura_referencia']) + 3)
    
    return cortes