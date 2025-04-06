from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename # For handling filenames securely
import os
import io

# Import the prediction function from our inference script
from inference import predict_pose, loaded_model

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handles image upload and returns pose prediction."""
    if loaded_model is None:
         return jsonify({'error': 'Model not loaded on server. Cannot perform prediction.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Read image bytes directly from the file stream
            img_bytes = file.read()
            
            # Get prediction
            quaternion, translation = predict_pose(img_bytes)
            
            if quaternion is not None and translation is not None:
                # Format the results nicely
                result = {
                    'quaternion': [round(x, 5) for x in quaternion],
                    'translation': [round(x, 5) for x in translation]
                }
                return jsonify(result)
            else:
                return jsonify({'error': 'Prediction failed on server.'}), 500
                
        except Exception as e:
            print(f"Error processing prediction request: {e}")
            # Log the exception details in a real app
            return jsonify({'error': 'An internal error occurred.'}), 500
    else:
        return jsonify({'error': 'File type not allowed.'}), 400

# Serve static files (CSS, JS)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Use environment variables for host and port, useful for deployment
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    # Debug should be False in production
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true' 
    print(f"Starting Flask server on {host}:{port} with debug={debug}")
    app.run(host=host, port=port, debug=debug)
