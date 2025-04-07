import { vec3, quat } from 'https://cdn.skypack.dev/gl-matrix';

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const predictButton = document.getElementById('predictButton');
    const imagePreview = document.getElementById('imagePreview');
    const predictionResultDiv = document.getElementById('predictionResult');
    const statusDiv = document.getElementById('status');
    const poseInfoDiv = document.getElementById('poseInfo');
    const modelViewerElement = document.getElementById('modelViewer'); // Added ID to model-viewer in HTML
    const visualizationImage = document.getElementById('visualizationImage'); // Get visualization image element

    let selectedFile = null;
    let currentBlobUrl = null; // To keep track for potential cleanup

    // Function to convert base64 to Blob
    function base64ToBlob(base64, contentType = '', sliceSize = 512) {
        const byteCharacters = atob(base64);
        const byteArrays = [];
        for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
            const slice = byteCharacters.slice(offset, offset + sliceSize);
            const byteNumbers = new Array(slice.length);
            for (let i = 0; i < slice.length; i++) {
                byteNumbers[i] = slice.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            byteArrays.push(byteArray);
        }
        return new Blob(byteArrays, { type: contentType });
    }

    // --- Coordinate System Transformation ---
    // Rotation to convert from Camera coordinates (Z forward, Y down)
    // to ModelViewer coordinates (Y up, Z out) is typically 180 degrees around X-axis.
    const qCvToMv = quat.create();
    quat.setAxisAngle(qCvToMv, [1, 0, 0], Math.PI); // 180 degrees around X

    function transformPoseCvToMv(qCv, tCv) {
        // Input: qCv = [w, x, y, z], tCv = [x, y, z] from backend prediction
        // Output: { qMv: [x, y, z, w], tMv: [x, y, z] } for ModelViewer

        // 1. Convert input quaternion (assuming w,x,y,z) to gl-matrix format (x,y,z,w)
        const qCvGlm = quat.fromValues(qCv[1], qCv[2], qCv[3], qCv[0]);

        // 2. Apply coordinate system rotation
        const qMvGlm = quat.create();
        quat.multiply(qMvGlm, qCvToMv, qCvGlm);
        quat.normalize(qMvGlm, qMvGlm); // Ensure it's a unit quaternion

        // 3. Transform translation vector
        const tCvVec3 = vec3.fromValues(tCv[0], tCv[1], tCv[2]);
        const tMvVec3 = vec3.create();
        // Apply coordinate system change: (x, y, z)_cv -> (x, -y, -z)_mv
        vec3.set(tMvVec3, tCvVec3[0], -tCvVec3[1], -tCvVec3[2]);

        // Return in formats needed by model-viewer API
        return {
            qMvString: `${qMvGlm[0]} ${qMvGlm[1]} ${qMvGlm[2]} ${qMvGlm[3]}`, // x y z w format
            tMvString: `${tMvVec3[0]} ${tMvVec3[1]} ${tMvVec3[2]}`,       // x y z format
            tMvArray: [tMvVec3[0], tMvVec3[1], tMvVec3[2]] // For camera target
        };
    }
    // --- End Coordinate System Transformation ---


    fileInput.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            }
            reader.readAsDataURL(selectedFile);
            predictButton.disabled = false;
            setStatus('');
            predictionResultDiv.innerHTML = ''; // Clear model viewer area too
            modelViewerElement.src = null;
            modelViewerElement.style.display = 'none';
            visualizationImage.src = '#'; // Clear visualization
            visualizationImage.style.display = 'none';
            poseInfoDiv.innerHTML = '<p>Image selected. Click \'Predict & Generate\'.</p>'; // Inform user

            if (currentBlobUrl) {
                 URL.revokeObjectURL(currentBlobUrl); // Clean up old blob URL
                 currentBlobUrl = null;
            }
        } else {
            imagePreview.style.display = 'none';
            imagePreview.src = '#';
            predictButton.disabled = true;
            selectedFile = null;
            predictionResultDiv.innerHTML = '';
            modelViewerElement.src = null;
            modelViewerElement.style.display = 'none';
            visualizationImage.src = '#';
            visualizationImage.style.display = 'none';
            poseInfoDiv.innerHTML = '<p>Upload an image and click \'Predict & Generate\'.</p>';

             if (currentBlobUrl) {
                 URL.revokeObjectURL(currentBlobUrl);
                 currentBlobUrl = null;
            }
        }
    });

    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            setStatus('Please select an image file first.', 'error');
            return;
        }

        predictButton.disabled = true;
        setStatus('Processing... Predicting pose and generating mesh...', 'loading');
        predictionResultDiv.innerHTML = ''; // Clear results div for model viewer
        poseInfoDiv.innerHTML = '';
        modelViewerElement.style.display = 'none'; // Hide until loaded
        modelViewerElement.src = null;
        visualizationImage.src = '#'; // Clear previous visualization
        visualizationImage.style.display = 'none';

         if (currentBlobUrl) {
             URL.revokeObjectURL(currentBlobUrl);
             currentBlobUrl = null;
        }


        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                let finalStatus = 'Processing completed.';
                let statusType = 'success';

                // Display Pose Info (always try to display if available)
                if (result.quaternion && result.translation) {
                     poseInfoDiv.innerHTML = `
                        <p><strong>Predicted Pose (Camera Coordinates):</strong></p>
                        <p>Quaternion (w,x,y,z): [${result.quaternion.join(', ')}]</p>
                        <p>Translation (x,y,z): [${result.translation.join(', ')}]</p>
                     `;
                } else {
                     poseInfoDiv.innerHTML = '<p>Pose prediction data missing.</p>';
                }

                 // Display Visualization Image
                if (result.visualization_img_base64) {
                    visualizationImage.src = `data:image/png;base64,${result.visualization_img_base64}`;
                    visualizationImage.style.display = 'block';
                } else {
                    visualizationImage.src = '#';
                    visualizationImage.style.display = 'none';
                }

                // Handle mesh data - Create Blob URL and display model-viewer
                if (result.mesh_glb_base64 && result.quaternion && result.translation) {
                    console.log("Received 3D mesh and pose data. Setting up model-viewer...");
                    try {
                        const blob = base64ToBlob(result.mesh_glb_base64, 'model/gltf-binary');
                        currentBlobUrl = URL.createObjectURL(blob); // Store for cleanup

                        // Transform pose for model-viewer
                        const { qMvString, tMvString, tMvArray } = transformPoseCvToMv(result.quaternion, result.translation);
                        console.log("Transformed Pose (ModelViewer Coords):");
                        console.log("  Position:", tMvString);
                        console.log("  Orientation:", qMvString);


                        // Set the src *first*, then wait for load event
                        modelViewerElement.src = currentBlobUrl;
                        modelViewerElement.style.display = 'block'; // Show viewer area

                        // Add error handling for model loading
                        modelViewerElement.addEventListener('error', (event) => {
                            console.error('Model Viewer Error:', event.detail);
                            predictionResultDiv.innerHTML = `<p>Error loading 3D model: ${event.detail?.message || 'Unknown loading error'}. Check console.</p>`;
                            setStatus('Failed to load 3D model.', 'error');
                            modelViewerElement.style.display = 'none'; // Hide viewer on error
                         }, { once: true });

                        modelViewerElement.addEventListener('load', () => {
                             console.log("Model loaded, applying pose...");

                            // --- Apply Pose ---
                            modelViewerElement.orientation = qMvString;
                            modelViewerElement.position = tMvString;

                            // --- Adjust Camera ---
                            // Target the model's new position
                            modelViewerElement.cameraTarget = tMvString;
                            // Set a reasonable camera orbit (distance slightly away from model)
                            // Adjust '1.5m' based on typical model scale if needed
                            const distance = vec3.length(tMvArray) * 2.5; // Example: distance based on translation magnitude
                            modelViewerElement.cameraOrbit = `0deg 75deg ${Math.max(distance, 0.5)}m`; // Ensure minimum distance

                            modelViewerElement.jumpCameraToGoal(); // Apply camera changes immediately

                            console.log("Pose applied to model-viewer.");
                         }, { once: true }); // Important: use 'once' so listener is removed after firing

                        finalStatus = 'Pose prediction and mesh generation successful! Model posed.';
                        setStatus(finalStatus, statusType);

                    } catch (e) {
                         console.error("Error processing/posing mesh data:", e);
                         predictionResultDiv.innerHTML = `<p>Error displaying/posing 3D model: ${e.message}</p>`; // Show error in results div
                         modelViewerElement.style.display = 'none'; // Hide viewer on error
                         finalStatus = 'Pose prediction successful, but failed to display/pose mesh.';
                         setStatus(finalStatus, 'error');
                    }
                } else if (result.mesh_glb_base64) {
                     // Mesh received but pose missing
                     console.log("Mesh data received, but pose data missing. Displaying model without pose.");
                      const blob = base64ToBlob(result.mesh_glb_base64, 'model/gltf-binary');
                      currentBlobUrl = URL.createObjectURL(blob);
                      modelViewerElement.src = currentBlobUrl;
                      modelViewerElement.orientation = "0 0 0 1"; // Reset orientation
                      modelViewerElement.position = "0 0 0";   // Reset position
                      modelViewerElement.cameraTarget = "auto auto auto";
                      modelViewerElement.cameraOrbit = "auto auto auto";
                      modelViewerElement.style.display = 'block';
                      finalStatus = 'Mesh generated, but pose data missing. Displaying default view.';
                      setStatus(finalStatus, 'warning');
                }
                 else {
                    // No mesh generated or sent
                    console.log("Mesh generation skipped or failed.");
                    // Don't clear predictionResultDiv if pose info is there
                    // predictionResultDiv.innerHTML = '<p>Mesh generation failed or was skipped.</p>';
                     finalStatus = 'Pose prediction successful (mesh failed/skipped).';
                      setStatus(finalStatus, statusType); // Use statusType from initial check
                }

            } else {
                predictionResultDiv.innerHTML = `<p>Error: ${result.error || 'Unknown error'}</p>`;
                poseInfoDiv.innerHTML = '';
                setStatus('Processing failed.', 'error');
                 modelViewerElement.style.display = 'none';
                 visualizationImage.src = '#'; // Clear visualization on error
                 visualizationImage.style.display = 'none';
            }
        } catch (error) {
            console.error('Prediction request failed:', error);
            predictionResultDiv.innerHTML = `<p>Error: Could not connect to the server or an unexpected error occurred.</p>`;
            poseInfoDiv.innerHTML = '';
            setStatus('Request failed.', 'error');
             modelViewerElement.style.display = 'none';
             visualizationImage.src = '#'; // Clear visualization on error
             visualizationImage.style.display = 'none';
        } finally {
            if (selectedFile) {
                predictButton.disabled = false;
            }
        }
    });

    function setStatus(message, type = 'info') {
        statusDiv.textContent = message;
        statusDiv.className = `status-message ${type}`;
    }
});