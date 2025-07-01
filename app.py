# from flask import Flask, render_template, request
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model = load_model('MakananTradisional.keras')

# # Harus sesuai urutan pelatihan
# class_names = [
#     "makanan___rendang", "makanan___sate", "makanan___bakso", "makanan___gado_gado", "makanan___gudeg"
# ]

# # Resep per label
# recipe_map = {
#     "makanan___rendang": "1. Daging sapi, santan, serai, daun jeruk...\n2. Masak 2–3 jam hingga kering.",
#     "makanan___sate": "1. Daging ayam, tusuk sate, bumbu kacang...\n2. Bakar sampai matang.",
#     "makanan___bakso": "1. Daging giling, tepung tapioka, bawang...\n2. Rebus hingga mengapung.",
#     "makanan___gado_gado": "1. Sayuran rebus, tahu, tempe, bumbu kacang...\n2. Sajikan dengan lontong.",
#     "makanan___gudeg": "1. Nangka muda, santan, telur, daun jati...\n2. Masak lama hingga pekat."
# }

# def predict_image(img_path):
#     image = load_img(img_path, target_size=(64, 64))
#     image = img_to_array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     predictions = model.predict(image)
#     class_index = np.argmax(predictions)
#     confidence = float(np.max(predictions))
#     label = class_names[class_index]
#     recipe = recipe_map.get(label, "Resep tidak tersedia.")
#     return label, confidence, recipe

# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route("/deteksi", methods=["GET", "POST"])
# def deteksi():
#     if request.method == "POST":
#         file = request.files["image"]
#         if file:
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)

#             label, confidence, recipe = predict_image(file_path)
#             return render_template("index.html", label=label, confidence=confidence, image_url=file_path, recipe=recipe)

#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
import os
import numpy as np
import base64
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

model = load_model('MakananTradisional.keras')

class_names = [
    "makanan___rendang", "makanan___sate", "makanan___bakso", "makanan___gado_gado", "makanan___gudeg"
]

recipe_map = {
    "makanan___rendang": "1. Daging sapi, santan, serai, daun jeruk...\n2. Masak 2–3 jam hingga kering.",
    "makanan___sate": "1. Daging ayam, tusuk sate, bumbu kacang...\n2. Bakar sampai matang.",
    "makanan___bakso": "1. Daging giling, tepung tapioka, bawang...\n2. Rebus hingga mengapung.",
    "makanan___gado_gado": "1. Sayuran rebus, tahu, tempe, bumbu kacang...\n2. Sajikan dengan lontong.",
    "makanan___gudeg": "1. Nangka muda, santan, telur, daun jati...\n2. Masak lama hingga pekat."
}

def predict_from_base64(base64_data):
    try:
        # Decode base64 menjadi gambar
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((64, 64))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Prediksi
        predictions = model.predict(image)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        label = class_names[class_index]
        recipe = recipe_map.get(label, "Resep tidak tersedia.")

        return label, confidence, recipe
    except Exception as e:
        return None, None, f"Error during prediction: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400

    label, confidence, recipe = predict_from_base64(data['image'])
    if label is None:
        return jsonify({"error": recipe}), 500

    return jsonify({
        "label": label,
        "confidence": confidence,
        "recipe": recipe
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
