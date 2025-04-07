# Script for loading the mode
from flask import Flask
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from vision_api import create_routes


model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

app = Flask(__name__)

create_routes(app, model, processor)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

