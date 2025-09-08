FROM python:3.10-slim

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

CMD ["python3", "app.py"]