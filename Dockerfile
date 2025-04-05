# Use an official PyTorch image with CUDA support as a parent image
# Adjust the tag based on your required PyTorch and CUDA versions
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file first to leverage Docker cache
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY train.py ./

# Make port 80 available to the world outside this container (optional, not needed for training)
# EXPOSE 80 

# Define environment variables (optional)
# ENV NAME World

# Run train.py when the container launches
# The script will be executed with any arguments passed to `docker run`
ENTRYPOINT ["python", "train.py"]

# Example of default command-line arguments (optional)
# You will typically override these when running the container
# CMD ["--annotation-file", "/data/train.json", "--img-dir", "/data/images/train", ...] 