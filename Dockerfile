# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit uses (default is 8501)
EXPOSE 8501

# Optional: Set environment variables for Streamlit (disable CORS if needed)
ENV STREAMLIT_SERVER_ENABLECORS=false

# Run the Streamlit app on port 8501 and bind to all network interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
