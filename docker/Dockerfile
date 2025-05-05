FROM python:3.11

RUN apt-get update && apt-get install build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /fastapi

COPY ../requirements.txt /fastapi/requirements.txt

RUN pip install --no-cache-dir -r /fastapi/requirements.txt

COPY .. /fastapi/

RUN pip install --no-cache-dir /fastapi/

CMD ["python", "/fastapi/main.py"]

