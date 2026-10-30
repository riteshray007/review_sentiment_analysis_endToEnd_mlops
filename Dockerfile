# 1. Base image matching training runtime (Python 3.13)
FROM python:3.13-slim

# Prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 2. Ensure setuptools is available for editable install without network isolation
RUN pip install --no-cache-dir --upgrade pip setuptools

# 3. Cache dependencies layer
COPY flask-app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy configurations, source package, and app files (trailing slashes enforced)
COPY pyproject.toml params.yaml ./
COPY src/ ./src/
COPY flask-app/ ./

# 5. Fast, offline editable install without touching PyPI or isolated envs
RUN pip install --no-cache-dir --no-build-isolation --no-deps -e .

EXPOSE 5000

# Option A: Local testing with Flask development server 
# CMD ["python", "app.py"]

# Option B: Production WSGI server (active by default)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]