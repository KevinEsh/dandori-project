FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip & pip install pipenv

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


COPY solution.py solution.py

ENTRYPOINT [ "python", "solution.py", "2023", "5", "model.bin"]