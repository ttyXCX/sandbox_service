import requests
import time

URL = "http://0.0.0.0:8761/vision/detect_objects"

if __name__ == "__main__":
    with open("/data/cty/sandbox_service/web_vision/desk_encoded.txt", "r") as f:
        img_str = f.read()
    
    data = {
        "image": img_str
    }
    
    time_start = time.time()
    for i in range(100):
        print("requesting... {}/{} {}s/per req".format(i + 1, 100, (time.time() - time_start) / (i + 1)))
        rsp = requests.post(url=URL, data=data)
    time_end = time.time()
    print("Time consumed {}s".format(time_end - time_start))