import numpy as np
import cv2
from ultralytics import YOLO
# import torch
# torch.multiprocessing.set_start_method("spawn")

# YOLO_WEIGHT = "/data/cty/sandbox_service/web_vision/models/weights/yolov8n.pt"
YOLO_WEIGHT = "/data/cty/sandbox_service/web_vision/models/weights/yolov8n.engine" # light-weight
# HSV阈值控制
SATURATION_LOWER_RATIO: float = 1.3  # 平均饱和度乘以此系数，作为饱和度下限，推荐值1.3
VALUE_LOWER: int = 140  # 明度下限
RED_LOWER: int = 150  # 红色色相下限
RED_UPPER: int = 180  # 红色色相上限
YELLOW_LOWER: int = 10  # 黄色色相下限
YELLOW_UPPER: int = 60  # 黄色色相上限
GREEN_LOWER: int = 70  # 绿色色相下限
GREEN_UPPER: int = 100  # 绿色色相上限


class VisionDetection(object):
    def __init__(self, weight,
                 save_img=False):
        self.model = YOLO(weight, task="detect")
        # self.OBJECT_CLS = self.model.names
        # print(self.model.info())
        
        self.save_img = save_img
    
    def predict_from_path(self, img_path):
        results = self.model.predict(source=img_path, 
                                     device=0, half=True,
                                     save=self.save_img)
        return results
    
    def predict_from_array(self, img_arr):
        results = self.model.predict(source=img_arr, 
                                     device=0, half=True,
                                     save=self.save_img)
        return results
    
    def parse_info(self, rs):
        info = {"speed": rs.speed,
                "image_shape": rs.orig_shape,
                "objects": []}
        
        names = rs.names
        boxes = rs.boxes
        
        l = boxes.shape[0]
        for i in range(l):
            obj_idx = int(boxes.cls[i])
            obj = {
                "name": names[obj_idx],
                "confidence": float(boxes.conf[i]),
                "box": {
                    "xyxy": boxes.xyxy[i].cpu().detach().numpy().tolist(),
                    "xywh": boxes.xywh[i].cpu().detach().numpy().tolist(),
                    "xyxyn": boxes.xyxyn[i].cpu().detach().numpy().tolist(),
                    "xywhn": boxes.xywhn[i].cpu().detach().numpy().tolist(),
                }
            }
            info["objects"].append(obj)
        
        return info


class TrafficLightDetection(VisionDetection):
    def __init__(self, vision_weight,
                 save_img=False) -> None:
        super(TrafficLightDetection, self).__init__(vision_weight, save_img)
        self.OBJECT_CLS = None

    def __crop_traffic_light(self, img, boxes):
        box_pos = boxes.xyxy
        box_cls = boxes.cls
        img_traffic_light = None
        
        for i in range(len(box_cls)):
            cls = int(box_cls[i])
            if self.OBJECT_CLS[cls] != "traffic light": # filter
                continue
            
            pos = box_pos[i]
            x1, y1, x2, y2 = int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])
            img_traffic_light = img[y1: y2 + 1, x1: x2 + 1, :]
            
        return img_traffic_light
    
    @staticmethod
    def __recognize_color(img_light):
        if img_light is None:
            return None
        
        def get_avg_saturation(hsv_img):
            s_channel = hsv_img[: , :, 1]
            return np.average(s_channel)
        
        def apply_mask(rgb_img, hsv_img,
                       sat_lower, val_lower,
                       clr_lower, clr_upper):
            lower = np.array([clr_lower, sat_lower, val_lower])
            upper = np.array([clr_upper, 255, 255])
            clr_mask = cv2.inRange(hsv_img, lower, upper)
            
            masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=clr_mask)
            return masked_img

        def count_pixel(img):
            merge_channel = np.sum(img, axis=2)
            return np.sum(merge_channel != 0)

        # preprocess
        hsv_img = cv2.cvtColor(img_light, cv2.COLOR_BGR2HSV)
        
        avg_saturation = get_avg_saturation(img_light)
        sat_lower = int(avg_saturation * SATURATION_LOWER_RATIO)
        val_lower = VALUE_LOWER

        # filter color
        red_masked = apply_mask(img_light, hsv_img,
                                sat_lower, val_lower,
                                RED_LOWER, RED_UPPER)
        yellow_masked = apply_mask(img_light, hsv_img,
                                   sat_lower, val_lower,
                                   YELLOW_LOWER, YELLOW_UPPER)
        green_masked = apply_mask(img_light, hsv_img,
                                  sat_lower, val_lower,
                                  GREEN_LOWER, GREEN_UPPER)
        # count valid pixels
        red_cnt = count_pixel(red_masked)
        yellow_cnt = count_pixel(yellow_masked)
        green_cnt = count_pixel(green_masked)
        clr_cnt = max(red_cnt, yellow_cnt, green_cnt)
        
        # determine color
        color = None
        if clr_cnt == 0:
            color = None
        elif clr_cnt == red_cnt:
            color = "red"
        elif clr_cnt == yellow_cnt:
            color = "yellow"
        elif clr_cnt == green_cnt:
            color = "green"
        
        return color
    
    def detect_color_from_path(self, img_path):
        vision_results = self.predict_from_path(img_path)
        color = None
        
        for rs in vision_results:
            if self.OBJECT_CLS is None:
                self.OBJECT_CLS = rs.names
            
            boxes = rs.boxes
            img_traffic_light = self.__crop_traffic_light(rs.orig_img, boxes)
            color = self.__recognize_color(img_traffic_light)
            # one image at a time
            break
        
        return color
    
    def detect_color_from_array(self, img_arr):
        vision_results = self.predict_from_path(img_arr)
        color = None
        
        for rs in vision_results:
            if self.OBJECT_CLS is None:
                self.OBJECT_CLS = rs.names
                
            boxes = rs.boxes
            img_traffic_light = self.__crop_traffic_light(rs.orig_img, boxes)
            color = self.__recognize_color(img_traffic_light)
            # one image at a time
            break
        
        return color


if __name__ == "__main__":
    img_path = "/data/cty/sandbox_service/web_vision/rgb_test.png"

    vision_traffic = TrafficLightDetection(YOLO_WEIGHT)
    from convert import ImageConverter
    img_cvt = ImageConverter()
    img_str = img_cvt.encode_from_path(img_path)
    img_arr = img_cvt.decode(img_str)
    
    # with open("/data/cty/sandbox_service/web_vision/rgb_test.txt", "w") as f:
    #     f.write(img_str)
    
    results = vision_traffic.predict_from_array(img_arr)
    obj_info = vision_traffic.parse_info(results[0])
    
    for i in range(100):
        color = vision_traffic.detect_color_from_array(img_arr)
    #     color = vision_traffic.detect_color_from_path(img_path)