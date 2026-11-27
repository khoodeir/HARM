FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY haram_counter.py index.html streams.json ./
ENV HOST=0.0.0.0 PORT=8765
EXPOSE 8765
# مرّر رمز الإدارة: docker run -e HARAM_TOKEN=سر -p 8765:8765 haram
CMD ["python", "haram_counter.py"]
