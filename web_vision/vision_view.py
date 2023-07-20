import traceback
from flask import jsonify, request
from web_vision import vision_blueprint
from web_vision.models.detect import TrafficLightDetection, YOLO_WEIGHT
from web_vision.models.convert import ImageConverter

# init model
INIT_IMG_PATH = "/data/cty/sandbox_service/web_vision/rgb_test.png"
model = TrafficLightDetection(YOLO_WEIGHT)
# _ = model.predict_from_path(INIT_IMG_PATH)
# image processor
img_cvt = ImageConverter()


@vision_blueprint.route("/detect_objects", methods=["POST"])
def detect_objects():
    
    try:
        # get data
        if request.content_type.startswith("application/json"):
            img = request.json.get("image")
        elif request.content_type.startswith("multipart/form-data"):
            img = request.form.get("image")
        else:
            img = request.values.get("image")            
        
        img_arr = img_cvt.decode(img)
        if img_arr is None:
            response = {
                "status": -2,
                "data": None,
                "msg": "Invalid image argument"
            }
        else:
            results = model.predict_from_array(img_arr)
            response_data = model.parse_info(results[0])
            response = {
                "status": 200,
                "data": response_data,
                "msg": "success"
            }
    except Exception as e:
        traceback.print_exc()
        response = {
            "status": -1,
            "data": None,
            "msg": e.args
        }
    finally:
        return jsonify(response)


@vision_blueprint.route("/detect_traffic_light_color", methods=["POST"])
def detect_light_color():
    
    try:
        # get data
        if request.content_type.startswith("application/json"):
            img = request.json.get("image")
        elif request.content_type.startswith("multipart/form-data"):
            img = request.form.get("image")
        else:
            img = request.values.get("image")            
        
        img_arr = img_cvt.decode(img)
        if img_arr is None:
            response = {
                "status": -2,
                "data": None,
                "msg": "Invalid image argument"
            }
        else:
            color = model.detect_color_from_array(img_arr)
            response = {
                "status": 200,
                "data": {"color": color},
                "msg": "success"
            }
    except Exception as e:
        traceback.print_exc()
        response = {
            "status": -1,
            "data": None,
            "msg": e.args
        }
    finally:
        return jsonify(response)