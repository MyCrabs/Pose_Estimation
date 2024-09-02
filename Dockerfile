FROM ubuntu
WORKDIR /src

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    python3-pil \
    python3-numpy \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /src/venv

RUN . /src/venv/bin/activate && pip install --upgrade pip && pip install tensorflow gradio mediapipe

ENV PATH="/src/venv/bin:$PATH"

COPY main.py ./main.py
COPY NewModel.keras ./NewModel.keras

CMD ["python3", "main.py"]