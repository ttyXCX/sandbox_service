from flask import Blueprint

vision_blueprint = Blueprint("vision", __name__, url_prefix='/vision')

from web_vision import vision_view