FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg sox libsndfile1 build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git && \
    cd GPT-SoVITS && pip install --no-cache-dir -r requirements.txt

COPY pipeline/ /app/pipeline/
COPY agents/ /app/agents/
COPY patches/ /app/patches/
RUN python patches/apply_mac_fixes.py || true
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
RUN mkdir -p raw cleaned sliced dataset output logs

EXPOSE 7861 9880 8765
CMD ["python", "agents/support_agent.py"]
