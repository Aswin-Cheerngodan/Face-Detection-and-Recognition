# Use an official lightweight Python base image
FROM python:3.10-slim


# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy your project files into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "src/main.py", "--server.fileWatcherType", "none", "--server.port=8501", "--server.address=0.0.0.0"]
