FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY audio_server.py .

ENV GCP_PROJECT=ayusha-you2
ENV GCP_LOCATION=us-central1
ENV AUDIO_BUCKET=open-files-app
ENV GEMINI_MODEL=gemini-2.5-pro-preview-03-25
ENV IMAGE_MODEL_NANO=gemini-3.1-flash-image-preview
ENV IMAGE_MODEL_PRO=gemini-3-pro-image-preview

CMD ["python", "audio_server.py"]
