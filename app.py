import os

from flask import (
    Flask,
    render_template,
    request
)

from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No Image Uploaded"

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"], 
        file.filename 
    )

    file.save(filepath)

    gender, confidence = predict_image(filepath)

    return render_template(
        "result.html",
        gender=gender,
        confidence=confidence,
        image=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)