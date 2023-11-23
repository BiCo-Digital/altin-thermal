import requests
import datetime

# Set the URL of the server endpoint
url = "http://localhost:8888/api/event"

current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Set the form data and images
form_data = {
    "line": 2,
    "min_t": 25.4,
    "avg_t": 26.3,
    "delta_t": 0.9,
    "max_t": 27.2,
    "min_t_zscore": 0.1,
    "q_min_t": 25.4,
    "q_delta_t": 0.9,
    "timestamp": current_timestamp,
}

files = {
    "image_thermal": ('image_thermal.png', open('experiments/XXX/frame_06-26-54_4.png', "rb"), "image/png"),
    "image_color": ('image_color.png', open('experiments/thermal_images_1/frame_0002.png', "rb"), "image/png"),
}

# Send the POST request
response = requests.post(url, data=form_data, files=files)

# Check the response status code
if response.status_code == 200:
    print("Request successful!", response.text)
else:
    print("Request failed.")



import requests
import threading

def upload_to_server(line, min_t, avg_t, delta_t, max_t, min_t_zscore, q_min_t, q_delta_t, timestamp, image_thermal, image_color):
    form_data = {
        "line": line,
        "min_t": min_t,
        "avg_t": avg_t,
        "delta_t": delta_t,
        "max_t": max_t,
        "min_t_zscore": min_t_zscore,
        "q_min_t": q_min_t,
        "q_delta_t": q_delta_t,
        "timestamp": timestamp,
    }

    files = {
        "image_thermal": ('image_thermal.png', open(f'experiments/{image_thermal}', "rb"), "image/png"),
        "image_color": ('image_color.png', open(f'experiments/thermal_images_1/frame_0002.png', "rb"), "image/png"),
    }

    def send_request():
        response = requests.post("http://localhost:8888/api/event", data=form_data, files=files)
        if response.status_code == 200:
            print("Request successful!", response.text)
        else:
            print("Request failed.")

    thread = threading.Thread(target=send_request)
    thread.start()