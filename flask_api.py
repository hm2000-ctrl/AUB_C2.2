
from flask import request, jsonify, send_file
from PIL import Image
import torch
import numpy as np
import io

def get_colored_mask(predicted):
    
    colormap = np.array([
        [0, 0, 0],        
        [128, 0, 0],       
        [0, 128, 0],       
        [128, 128, 0],     
        [0, 0, 128],       
        [128, 0, 128],     
        [0, 128, 128],     
        [128, 128, 128],   
    ])
    color_mask = colormap[predicted % len(colormap)]
    return Image.fromarray(color_mask.astype(np.uint8))

def apply_overlay(original, mask):
    original = original.convert("RGBA")
    mask = mask.convert("RGBA")
    return Image.blend(original, mask, alpha=0.5)

def create_routes(app, model, processor):
    @app.route("/segment", methods=["POST"])
    def segment():
        if 'file' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['file']
        image = Image.open(file.stream).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted = torch.argmax(logits, dim=1)[0].cpu().numpy()
        
        
        mask = Image.fromarray(predicted.astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
        mask_colored = get_colored_mask(np.array(mask))

        
        result = apply_overlay(image, mask_colored)

        
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')

