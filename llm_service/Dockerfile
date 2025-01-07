FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python packages and running the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry config virtualenvs.create false

# Copy Poetry dependency files into the container
COPY ./pyproject.toml ./poetry.lock* /app/

# Install Python dependencies (Production or Development)
ARG INSTALL_DEV=true
RUN export PATH="/root/.local/bin:$PATH" && \
    if [ "$INSTALL_DEV" = "true" ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi

# Copy application source code into the container
COPY ./app /app/app

# Install Uvicorn and Gunicorn globally
RUN pip install --no-cache-dir uvicorn gunicorn

# Expose the application's default port
EXPOSE 8000

# Set environment variables
ENV VARIABLE_NAME=api
ENV PYTHONPATH=/app

# Command to run the FastAPI application using Gunicorn with Uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:api"]
