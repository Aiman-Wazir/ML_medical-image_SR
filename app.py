from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

mri_model = load_model("mri_unet_model.keras")
xray_model = load_model("XRAY_NEW.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    image_type = request.form["image_type"]
    file = request.files["image"]

    # Open image
    original = Image.open(file).convert("L")

    # ✅ Resize for DISPLAY (before image)
    original = original.resize((256, 256), Image.LANCZOS)
    original.save("static/original.png")

    # Copy for model input
    img = original.copy()

    # model preprocessing + prediction
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 256, 256, 1)

    if image_type == "mri":
        prediction = mri_model.predict(img)
    else:
        prediction = xray_model.predict(img)

    # convert prediction to image
    output = prediction[0].squeeze()
    output = (output * 255).astype(np.uint8)

    result = Image.fromarray(output)
    result.save("static/result.png")

    return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=True)
    