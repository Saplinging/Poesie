FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    make \
    python3 \
    python3-pip \
    python3-venv \
    python3-pygments \
    && rm -rf /var/lib/apt/lists/*

# Python venv (PEP 668 kompatibel)
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="${VENV_PATH}/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

CMD ["bash"]
