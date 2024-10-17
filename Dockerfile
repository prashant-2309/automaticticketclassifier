FROM python:3.9-slim
 
WORKDIR /app
 
# Copy model files and application files
COPY model.safetensors /app/model.safetensors
COPY app.py /app/app.py
 
# Install necessary libraries
RUN pip install torch transformers flask gunicorn
 
# Expose the port
EXPOSE 8080
 
# Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]