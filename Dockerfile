FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data_source.py predict.py ./
COPY artifacts/model.joblib artifacts/model.joblib
USER 10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
