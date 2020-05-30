import requests
import base64

URL = "http://127.0.0.1:5000/transcribe"

encode_string = base64.b64encode(open("audio.wav", "rb").read())
a = {"username": "firstuser",
    "audio_file": encode_string,
    "create_output_directory": 0}

r = requests.request(method='get', url=URL, data=a)