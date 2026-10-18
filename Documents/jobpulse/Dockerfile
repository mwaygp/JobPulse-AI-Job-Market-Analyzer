FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "jobpulse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
