from flask import Flask, request, jsonify
from speakerDiarization import server_entry_point
app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify({"about": "Hello World"})


@app.route('/transcribe', methods=['GET'])
def transcribe():
    print(request)
    input_file = request.form
    final_transcript = server_entry_point(input_file)
    return final_transcript


if __name__ == '__main__':
    app.run(debug=True)


