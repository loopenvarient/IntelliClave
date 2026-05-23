FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.23.2 \
    torch==2.1.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    redis==5.0.1

COPY config/                    ./config/
COPY fl/                        ./fl/
COPY security/                  ./security/
COPY tee/sealed_storage/        ./tee/sealed_storage/
COPY tee/tee_mode.py            ./tee/tee_mode.py
COPY dashboard/backend/main.py              ./dashboard/backend/main.py
COPY dashboard/backend/training_freshness.py ./dashboard/backend/training_freshness.py
COPY data/class_weights.json    ./data/class_weights.json

ENV PYTHONPATH=/app
ENV DASHBOARD_STATE_DIR=/app/results
EXPOSE 8001

WORKDIR /app/dashboard/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
