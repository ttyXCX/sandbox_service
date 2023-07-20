import base64
import cv2
import numpy as np


class ImageConverter(object):
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def encode_from_path(img_path):
        encoded_str = None
        with open(img_path, "rb") as img:
            encoded_str = base64.b64encode(img.read()).decode()
        return encoded_str
    
    @staticmethod
    def decode(img_str):
        img_byte = base64.b64decode(img_str)
        image = np.frombuffer(img_byte, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    
    @staticmethod
    def compress_image(img, ratio):
        return cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    

if __name__ == "__main__":
    img_path = "/data/cty/sandbox_service/web_vision/desk_compressed.jpg" # _compressed
    img_converter = ImageConverter()
    
    en = img_converter.encode_from_path(img_path)
    
    # # compression
    # img = img_converter.decode(en)
    
    # x, y, _ = img.shape
    # TARGET_PIXEL = 48
    # ratio = 48 / min(x, y)
    
    # compressed_img = img_converter.compress_image(img, ratio) #None, fx=ratio, fy=ratio)
    # cv2.imwrite("/data/cty/sandbox_service/web_vision/desk_compressed.jpg", compressed_img)
    
    with open("/data/cty/sandbox_service/web_vision/desk_compressed.txt", "w") as f:
        f.write(en)
    
    de = img_converter.decode(en)
    
    