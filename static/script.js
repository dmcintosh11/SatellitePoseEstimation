document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const predictButton = document.getElementById('predictButton');
    const imagePreview = document.getElementById('imagePreview');
    const predictionResultDiv = document.getElementById('predictionResult');
    const statusDiv = document.getElementById('status');
    const poseInfoDiv = document.getElementById('poseInfo');

    let selectedFile = null;

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

    fileInput.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            // Display image preview
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            }
            reader.readAsDataURL(selectedFile);

            // Enable predict button
            predictButton.disabled = false;
            setStatus(''); // Clear previous status
            predictionResultDiv.innerHTML = '<p>Image selected. Click \'Predict & Generate\'.</p>';
            poseInfoDiv.innerHTML = ''; // Clear pose info
        } else {
            // No file selected or selection cancelled
            imagePreview.style.display = 'none';
            imagePreview.src = '#';
            predictButton.disabled = true;
            selectedFile = null;
            predictionResultDiv.innerHTML = '<p>Upload an image and click \'Predict & Generate\'.</p>';
            poseInfoDiv.innerHTML = ''; // Clear pose info
        }
    });

    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            setStatus('Please select an image file first.', 'error');
            return;
        }

        // Disable button and show loading status
        predictButton.disabled = true;
        setStatus('Processing... Predicting pose and generating mesh (this may take a minute)...', 'loading');
        predictionResultDiv.innerHTML = '<p>Processing...</p>'; // Clear previous results
        poseInfoDiv.innerHTML = ''; // Clear pose info

        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Send request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
                // No 'Content-Type' header needed, browser sets it for FormData
            });

            const result = await response.json(); // Always expect JSON response

            if (response.ok) {
                let finalStatus = 'Processing completed.';
                let statusType = 'success';

                // Display Pose Info
                if (result.quaternion && result.translation) {
                     poseInfoDiv.innerHTML = `
                        <p><strong>Predicted Pose:</strong></p>
                        <p>Quaternion: [${result.quaternion.join(', ')}]</p>
                        <p>Translation: [${result.translation.join(', ')}]</p>
                     `;
                } else {
                     poseInfoDiv.innerHTML = '<p>Pose prediction data missing.</p>';
                }

                // Handle mesh data - Create Blob URL and display model-viewer
                if (result.mesh_glb_base64) {
                    console.log("Received 3D mesh data. Creating Blob URL...");
                    try {
                        const blob = base64ToBlob(result.mesh_glb_base64, 'model/gltf-binary');
                        const blobUrl = URL.createObjectURL(blob);
                        
                        predictionResultDiv.innerHTML = `
                            <model-viewer src="${blobUrl}" 
                                          alt="Generated 3D model of the satellite" 
                                          auto-rotate 
                                          camera-controls 
                                          style="width: 100%; height: 300px; background-color: #eee; border-radius: 5px;" 
                                          ar 
                                          ar-modes="webxr scene-viewer quick-look">
                                <div slot="progress-bar"></div> 
                            </model-viewer>
                        `;
                        // Note: Posing the model dynamically using result.quaternion/translation
                        // is complex and not directly supported by setting model-viewer attributes easily.
                        // It would typically require using the model-viewer API or a library like Three.js.
                        finalStatus = 'Pose prediction and mesh generation successful!';

                        // Clean up the Blob URL when it's no longer needed (e.g., when a new model is loaded)
                        // For simplicity, we don't implement explicit cleanup here, but it's good practice.
                        // URL.revokeObjectURL(blobUrl); 
                    } catch (e) {
                         console.error("Error processing mesh data:", e);
                         predictionResultDiv.innerHTML = `<p>Error displaying 3D model: ${e.message}</p>`;
                         finalStatus = 'Pose prediction successful, but failed to display mesh.';
                         statusType = 'error';
                    }
                } else {
                    console.log("Mesh generation skipped or failed.");
                    predictionResultDiv.innerHTML = '<p>Mesh generation failed or was skipped. Pose data above.</p>';
                     finalStatus = 'Pose prediction successful (mesh failed/skipped).';
                }
                setStatus(finalStatus, statusType);

            } else {
                // Display error from backend
                predictionResultDiv.innerHTML = `<p>Error: ${result.error || 'Unknown error'}</p>`;
                poseInfoDiv.innerHTML = ''; // Clear pose info on error
                setStatus('Processing failed.', 'error');
            }
        } catch (error) {
            console.error('Prediction request failed:', error);
            predictionResultDiv.innerHTML = `<p>Error: Could not connect to the server or an unexpected error occurred.</p>`;
            poseInfoDiv.innerHTML = ''; // Clear pose info on error
            setStatus('Request failed.', 'error');
        } finally {
            // Re-enable button if a file is still selected
            if (selectedFile) {
                predictButton.disabled = false;
            }
        }
    });

    function setStatus(message, type = 'info') {
        statusDiv.textContent = message;
        statusDiv.className = `status-message ${type}`; // Reset classes and add new type
    }
});