from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename # For handling filenames securely
import os
import io
import subprocess      # For running SF3D
import tempfile        # For temporary files
import base64          # For encoding mesh data
import glob            # For finding the output mesh file
import shutil          # For cleaning up directories

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

# Path to the stable-fast-3d run.py script
# IMPORTANT: Update this path to the correct location on your system!
SF3D_RUN_SCRIPT = os.path.expanduser('~/workspace/stable-fast-3d/run.py') 
if not os.path.exists(SF3D_RUN_SCRIPT):
    print(f"Warning: Stable Fast 3D run script not found at {SF3D_RUN_SCRIPT}. Mesh generation will fail.")
    # Consider raising an error or disabling the feature if critical

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handles image upload, predicts pose, generates 3D mesh, draws axes, and returns results."""
    if loaded_model is None:
         return jsonify({'error': 'Pose estimation model not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
        
    if file and allowed_file(file.filename):
        img_bytes = file.read()
        file.seek(0) # Reset stream
        base_filename = secure_filename(os.path.splitext(file.filename)[0]) # Get base name for output

        # --- Temporary file handling --- 
        temp_dir = None # Initialize
        output_mesh_path = None
        mesh_glb_base64 = None
        pose_quaternion = None
        pose_translation = None
        visualization_img_base64 = None

        try:
            # 1. Predict Pose
            print("Predicting pose...")
            pose_quaternion, pose_translation = predict_pose(img_bytes)
            if pose_quaternion is None or pose_translation is None:
                return jsonify({'error': 'Pose prediction failed on server.'}), 500
            print("Pose prediction successful.")

            # 2. Create temporary directory and save image for SF3D
            temp_dir = tempfile.mkdtemp(prefix='sf3d_proc_')
            temp_image_path = os.path.join(temp_dir, f"{base_filename}_input.png") # Use png for simplicity
            with open(temp_image_path, 'wb') as f_temp:
                f_temp.write(img_bytes)
            print(f"Saved temporary image to {temp_image_path}")

            # 3. Run Stable Fast 3D
            if not os.path.exists(SF3D_RUN_SCRIPT):
                 print(f"SF3D script not found at {SF3D_RUN_SCRIPT}, skipping mesh generation.")
                 # Optionally return an error or just skip
            else:
                print(f"Running Stable Fast 3D on {temp_image_path}...")
                sf3d_output_dir = os.path.join(temp_dir, 'sf3d_output')
                os.makedirs(sf3d_output_dir, exist_ok=True)
                
                # Construct command. Add any other desired SF3D args here.
                command = [
                    'python', 
                    SF3D_RUN_SCRIPT, 
                    temp_image_path, 
                    '--output-dir', sf3d_output_dir
                    # Add e.g. '--texture-resolution', '1024' if needed
                ]
                
                # Execute the command
                try:
                    # Capture output for debugging, set timeout? 
                    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300) # 5 min timeout 
                    print("SF3D stdout:", result.stdout)
                    print("SF3D stderr:", result.stderr)
                    print("Stable Fast 3D completed.")

                    # Find the generated .glb file (assuming one is created)
                    # SF3D might name it based on input, e.g., input_filename.glb
                    search_pattern = os.path.join(sf3d_output_dir, '*.glb')
                    glb_files = glob.glob(search_pattern)
                    if not glb_files:
                        print(f"Error: No .glb file found in {sf3d_output_dir}")
                        # Decide how to handle: error or proceed without mesh?
                    else:
                        output_mesh_path = glb_files[0] # Take the first one found
                        print(f"Found generated mesh: {output_mesh_path}")
                        # Read and encode the mesh file
                        with open(output_mesh_path, 'rb') as f_mesh:
                            mesh_glb_base64 = base64.b64encode(f_mesh.read()).decode('utf-8')
                        print("Encoded mesh to base64.")

                except subprocess.CalledProcessError as e:
                    print(f"Error running Stable Fast 3D: {e}")
                    print("SF3D stdout:", e.stdout)
                    print("SF3D stderr:", e.stderr)
                    # Return specific error or proceed without mesh?
                    return jsonify({'error': f'Stable Fast 3D execution failed: {e.stderr[:200]}...'}), 500
                except subprocess.TimeoutExpired:
                    print("Error: Stable Fast 3D timed out.")
                    return jsonify({'error': 'Mesh generation process timed out.'}), 500
                except Exception as e_inner:
                     print(f"An unexpected error occurred during SF3D processing: {e_inner}")
                     # May want to return 500

            # 4. Draw visualization axes (optional, if still desired)
            print("Generating pose visualization...")
            visualization_img_base64 = draw_pose_axes(img_bytes, pose_quaternion, pose_translation)
            if not visualization_img_base64:
                print("Warning: Failed to draw visualization axes.")
                # Handle error? Or just don't include it in response?
            else:
                 print("Visualization generated.")

            # 5. Prepare response
            response_data = {
                'quaternion': [round(x, 5) for x in pose_quaternion],
                'translation': [round(x, 5) for x in pose_translation],
                'mesh_glb_base64': mesh_glb_base64, # Will be None if SF3D failed/skipped
                'visualization_img_base64': visualization_img_base64 # May be None if drawing failed
            }
            return jsonify(response_data)
                
        except Exception as e:
            print(f"Error processing prediction request: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return jsonify({'error': f'An internal server error occurred: {str(e)[:100]}...'}), 500
        
        finally:
            # 6. Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e_cleanup:
                    print(f"Error cleaning up temp directory {temp_dir}: {e_cleanup}")

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
