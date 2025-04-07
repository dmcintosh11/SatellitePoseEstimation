from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename # For handling filenames securely
import os
import io

# Import the prediction function and the new drawing function
from inference import predict_pose, loaded_model
from visualization import draw_pose_axes # <-- Import drawing function

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
    """Handles image upload, predicts pose, draws axes, and returns visualized image."""
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
            # Need to reset stream pointer after reading if reading multiple times
            img_bytes = file.read()
            file.seek(0) # Reset stream pointer in case needed later (though not currently)
            
            # Get prediction
            quaternion, translation = predict_pose(img_bytes)
            
            if quaternion is not None and translation is not None:
                # Draw axes on the image
                # Pass the original image bytes again
                img_base64 = draw_pose_axes(img_bytes, quaternion, translation)

                if img_base64:
                    # Return the base64 encoded image with axes
                    return jsonify({'image_with_axes': img_base64}) # Return base64 image
                else:
                    # Error during drawing
                    return jsonify({'error': 'Prediction successful, but failed to draw visualization.'}), 500
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
    port = int(os.environ.get('FLASK_PORT', 8888))
    # Debug should be False in production
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true' 
    print(f"Starting Flask server on {host}:{port} with debug={debug}")
    app.run(host=host, port=port, debug=debug)
