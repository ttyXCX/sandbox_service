from flask import Flask
from flask_cors import *

from web_vision import vision_blueprint


def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    app.register_blueprint(vision_blueprint)

    return app
app = create_app()


# curl -H "Content-Type: application/json" -X POST  -d '{rmg image}' http://127.0.0.1:2300/vision/detect_objects

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8765)