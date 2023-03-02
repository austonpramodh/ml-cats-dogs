import validators
from flask import Blueprint, jsonify, request
import logging
from flask_expects_json import expects_json
from .models_apis import load_model, infer_image

bp = Blueprint('ingest', __name__, url_prefix='/cat_dog_v1')

schema = {
    "type": "object",
    "properties": {
        "url": {"type": "string"}
    },
    "required": ["url"]
}


@bp.route('/load-model', methods=['GET'])
def load_model_api():
    load_model()
    return jsonify({
        "success": True,
        "message": "Model Loaded!!"
    })


@bp.route('/detect-image', methods=['POST'])
@expects_json(schema, force=True)
def detect_img():
    # This is post request
    img_url = request.json["url"]
    if not validators.url(img_url):
        return jsonify({
            "success": False,
            "message": "Invalid URL given!"
        })

    # detect the classes, which one is this
    model_response = infer_image(img_url=img_url)

    data = {
        "success": True,
        "message": "Feteched a unique url with signature",
        "data": {
            "body": img_url,
            "model_response": model_response
        }
    }

    return jsonify(data)
