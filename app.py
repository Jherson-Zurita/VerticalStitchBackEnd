from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import uuid
from opencv_utils import procesar_video_completo, regenerar_imagen_nueva

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/procesar_video", methods=["POST"])
def procesar():
    if 'video' not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files['video']
    if not allowed_file(file.filename):
        return jsonify({"error": "Archivo no válido"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
    file.save(path)

    try:
        data = request.form
        img_path = procesar_video_completo(
            video_path=path,
            crop_percent=float(data.get("crop_percent", 0)),
            keep_first_original=data.get("keep_first_original", "true") == "true",
            keep_last_original=data.get("keep_last_original", "true") == "true",
            interval=int(data.get("interval", 1)),
            filtro=data.get("filtro", "canny"),
            metodo_comparacion=data.get("metodo_comparacion", "pixel"),
            umbral_coincidencia=float(data.get("umbral_coincidencia", 0.9)),
            incremento_paso=int(data.get("incremento_paso", 5)),
            escala_reduccion=float(data.get("escala_reduccion", 1.0)),
            metodo_fusion=data.get("metodo_fusion", "simple")
        )
        if os.path.exists(path):
            os.remove(path)
        return send_file(img_path, mimetype='image/jpeg')
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"error": str(e)}), 500

@app.route("/regenerar_imagen", methods=["POST"])
def regenerar():
    try:
        json_data = request.get_json()
        cortes = json_data.get("cortes", [])
        img_path = regenerar_imagen_nueva([float(x) for x in cortes])
        return send_file(img_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

