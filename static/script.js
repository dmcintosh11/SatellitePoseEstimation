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
    const exampleSelect = document.getElementById('exampleSelect'); // Get example dropdown

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

    // --- Fetch and Populate Examples ---
    async function loadExamples() {
        try {
            const response = await fetch('/examples');
            if (!response.ok) {
                console.error('Failed to fetch examples:', response.statusText);
                exampleSelect.disabled = true; // Disable dropdown on error
                exampleSelect.innerHTML = '<option value="">Examples unavailable</option>';
                return;
            }
            const data = await response.json();
            if (data.examples && data.examples.length > 0) {
                exampleSelect.innerHTML = '<option value="">-- Select --</option>'; // Reset
                data.examples.forEach(filename => {
                    const option = document.createElement('option');
                    option.value = filename;
                    option.textContent = filename;
                    exampleSelect.appendChild(option);
                });
                exampleSelect.disabled = false;
            } else {
                 exampleSelect.disabled = true;
                 exampleSelect.innerHTML = '<option value="">No examples found</option>';
            }
        } catch (error) {
            console.error('Error loading examples:', error);
            exampleSelect.disabled = true;
            exampleSelect.innerHTML = '<option value="">Error loading</option>';
        }
    }

    // --- Function to handle setting the selected file and updating UI ---
    function handleFileSelection(file) {
        selectedFile = file;
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            }
            reader.readAsDataURL(selectedFile);
            predictButton.disabled = false;
            setStatus('');
            modelViewerElement.src = null;
            modelViewerElement.style.display = 'none';
            visualizationImage.src = '#';
            visualizationImage.style.display = 'none';
            poseInfoDiv.innerHTML = '<p>Image selected. Click \'Predict & Generate\'.</p>';

            if (currentBlobUrl) {
                 URL.revokeObjectURL(currentBlobUrl);
                 currentBlobUrl = null;
            }
        } else {
             // Reset UI if file is null
             imagePreview.style.display = 'none';
             imagePreview.src = '#';
             predictButton.disabled = true;
             setStatus('');
             modelViewerElement.src = null;
             modelViewerElement.style.display = 'none';
             visualizationImage.src = '#';
             visualizationImage.style.display = 'none';
             poseInfoDiv.innerHTML = '<p>Upload an image or select an example.</p>';
              if (currentBlobUrl) {
                 URL.revokeObjectURL(currentBlobUrl);
                 currentBlobUrl = null;
             }
        }
    }

    // --- Event Listeners ---
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        handleFileSelection(file);
        exampleSelect.value = ""; // Deselect example dropdown if file is chosen
    });

    exampleSelect.addEventListener('change', async (event) => {
        const filename = event.target.value;
        if (filename) {
            setStatus('Loading example image...', 'loading');
            try {
                const response = await fetch(`/examples/${filename}`);
                if (!response.ok) {
                    setStatus(`Error loading example: ${response.statusText}`, 'error');
                    handleFileSelection(null); // Clear selection
                    return;
                }
                const imageBlob = await response.blob();
                // Create a File object from the Blob to mimic user upload
                const imageFile = new File([imageBlob], filename, { type: imageBlob.type });
                handleFileSelection(imageFile);
                fileInput.value = null; // Clear file input selection
                setStatus('Example image loaded.', 'success');

            } catch (error) {
                console.error('Error fetching example image:', error);
                setStatus('Failed to load example image.', 'error');
                handleFileSelection(null);
            }
        } else {
            // "-- Select --" chosen
            handleFileSelection(null);
        }
    });

    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            setStatus('Please select an image file or an example first.', 'error');
            return;
        }

        predictButton.disabled = true;
        setStatus('Processing... Predicting pose and generating mesh...', 'loading');
        poseInfoDiv.innerHTML = '';
        modelViewerElement.style.display = 'none';
        modelViewerElement.src = null;
        visualizationImage.src = '#';
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

                // Display Pose Info
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

                // Handle mesh data
                if (result.mesh_glb_base64 && result.quaternion && result.translation) {
                    console.log("Received 3D mesh and pose data. Setting up model-viewer...");
                    try {
                        const blob = base64ToBlob(result.mesh_glb_base64, 'model/gltf-binary');
                        currentBlobUrl = URL.createObjectURL(blob);

                        const { qMvString, tMvString, tMvArray } = transformPoseCvToMv(result.quaternion, result.translation);
                        console.log("Transformed Pose (ModelViewer Coords):");
                        console.log("  Position:", tMvString);
                        console.log("  Orientation:", qMvString);

                        modelViewerElement.src = currentBlobUrl;
                        modelViewerElement.style.display = 'block';

                        modelViewerElement.addEventListener('error', (event) => {
                            console.error('Model Viewer Error:', event.detail);
                            poseInfoDiv.innerHTML += `<p style="color: red;">Error loading 3D model: ${event.detail?.message || 'Unknown loading error'}.</p>`;
                            setStatus('Failed to load 3D model.', 'error');
                            modelViewerElement.style.display = 'none';
                         }, { once: true });

                        modelViewerElement.addEventListener('load', () => {
                             console.log("Model loaded, applying pose...");
                            modelViewerElement.orientation = qMvString;
                            modelViewerElement.position = tMvString;
                            modelViewerElement.cameraTarget = tMvString;
                            const distance = vec3.length(tMvArray) * 2.5;
                            modelViewerElement.cameraOrbit = `0deg 75deg ${Math.max(distance, 0.5)}m`;
                            modelViewerElement.jumpCameraToGoal();
                            console.log("Pose applied to model-viewer.");
                         }, { once: true });

                        finalStatus = 'Pose prediction and mesh generation successful! Model posed.';
                        setStatus(finalStatus, statusType);

                    } catch (e) {
                         console.error("Error processing/posing mesh data:", e);
                         poseInfoDiv.innerHTML += `<p style="color: red;">Error displaying/posing 3D model: ${e.message}</p>`;
                         modelViewerElement.style.display = 'none';
                         finalStatus = 'Pose prediction successful, but failed to display/pose mesh.';
                         setStatus(finalStatus, 'error');
                    }
                } else if (result.mesh_glb_base64) {
                     console.log("Mesh data received, but pose data missing. Displaying model without pose.");
                      const blob = base64ToBlob(result.mesh_glb_base64, 'model/gltf-binary');
                      currentBlobUrl = URL.createObjectURL(blob);
                      modelViewerElement.src = currentBlobUrl;
                      modelViewerElement.orientation = "0 0 0 1";
                      modelViewerElement.position = "0 0 0";
                      modelViewerElement.cameraTarget = "auto auto auto";
                      modelViewerElement.cameraOrbit = "auto auto auto";
                      modelViewerElement.style.display = 'block';
                      finalStatus = 'Mesh generated, but pose data missing. Displaying default view.';
                      setStatus(finalStatus, 'warning');
                }
                 else {
                    console.log("Mesh generation skipped or failed.");
                    poseInfoDiv.innerHTML += '<p>Mesh generation failed or was skipped.</p>';
                    finalStatus = 'Pose prediction successful (mesh failed/skipped).';
                    setStatus(finalStatus, statusType);
                }

            } else {
                poseInfoDiv.innerHTML = `<p style="color: red;">Error: ${result.error || 'Unknown error'}</p>`;
                setStatus('Processing failed.', 'error');
                 modelViewerElement.style.display = 'none';
                 visualizationImage.src = '#';
                 visualizationImage.style.display = 'none';
            }
        } catch (error) {
            console.error('Prediction request failed:', error);
            poseInfoDiv.innerHTML = `<p style="color: red;">Error: Could not connect to the server or an unexpected error occurred.</p>`;
            setStatus('Request failed.', 'error');
             modelViewerElement.style.display = 'none';
             visualizationImage.src = '#';
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

    // --- Initial Load ---
    loadExamples(); // Load examples when the page loads
});