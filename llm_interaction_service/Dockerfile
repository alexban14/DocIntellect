FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages and running the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1-mesa-glx \
    libpq-dev \
    autoconf \
    gcc \
    make && \
    apt-get -y install tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry globally
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry config virtualenvs.create false && \
    poetry config installer.max-workers 1

# Copy dependency files first (to leverage Docker cache)
COPY ./pyproject.toml ./poetry.lock* /app/

# Install Python dependencies (optimized for production or development)
RUN export PATH="/root/.local/bin:$PATH" && \
    if [ "$ENV" = "development" ] ; then poetry install --no-root --with dev; else poetry install --no-root --without dev; fi

# Copy application source code into the container
COPY ./app /app/app

# Install Uvicorn and Gunicorn globally
RUN pip install --no-cache-dir uvicorn gunicorn

# Expose application port
EXPOSE 8000

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH=/app
ENV VARIABLE_NAME=api

# Add and set up the entrypoint script
COPY ./.docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Remove unnecessary build dependencies for smaller images
RUN apt-get purge -y build-essential && apt-get autoremove -y

# Entrypoint command
ENTRYPOINT ["/app/entrypoint.sh"]
