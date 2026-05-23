FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY app ./app
COPY docs ./docs
COPY src ./src
COPY artifacts/reports ./artifacts/reports

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
