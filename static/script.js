document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const predictButton = document.getElementById('predictButton');
    const imagePreview = document.getElementById('imagePreview');
    const predictionResultDiv = document.getElementById('predictionResult');
    const statusDiv = document.getElementById('status');

    let selectedFile = null;

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
            predictionResultDiv.innerHTML = '<p>Image selected. Click \'Predict Pose\'.</p>';
        } else {
            // No file selected or selection cancelled
            imagePreview.style.display = 'none';
            imagePreview.src = '#';
            predictButton.disabled = true;
            selectedFile = null;
            predictionResultDiv.innerHTML = '<p>Upload an image and click \'Predict Pose\'.</p>';
        }
    });

    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            setStatus('Please select an image file first.', 'error');
            return;
        }

        // Disable button and show loading status
        predictButton.disabled = true;
        setStatus('Uploading and predicting...', 'loading');
        predictionResultDiv.innerHTML = ''; // Clear previous results

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
                // Display successful prediction - expect image data
                if (result.image_with_axes) {
                    predictionResultDiv.innerHTML = `
                        <img src="data:image/png;base64,${result.image_with_axes}" alt="Image with predicted pose axes" style="max-width: 100%; height: auto;">
                    `;
                    setStatus('Prediction and visualization successful!', 'success');
                } else {
                    // Handle case where server might return success but no image (shouldn't happen with current app.py)
                    predictionResultDiv.innerHTML = `<p>Error: Received success status but no visualized image data.</p>`;
                    setStatus('Visualization failed on server.', 'error');
                }
            } else {
                // Display error from backend
                predictionResultDiv.innerHTML = `<p>Error: ${result.error || 'Unknown error'}</p>`;
                setStatus('Prediction failed.', 'error');
            }
        } catch (error) {
            console.error('Prediction request failed:', error);
            predictionResultDiv.innerHTML = `<p>Error: Could not connect to the server or an unexpected error occurred.</p>`;
            setStatus('Prediction request failed.', 'error');
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