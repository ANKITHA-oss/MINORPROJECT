from flask import Flask, request, jsonify
from voice_security_module import VoiceSecurityModule
import os

app = Flask(__name__)
vsm = VoiceSecurityModule()

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        fmt = file.filename.split('.')[-1]
        audio_bytes = file.read()
        
        result = vsm.process_audio_and_return_result(audio_bytes, fmt)
        print("Processed:", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
