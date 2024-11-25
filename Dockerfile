FROM python:3.11-slim

RUN pip install pipenv gunicorn

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
# D:\Data\PyCharmProjects\mlzoomcamp-certification

RUN pipenv install --system --deploy
#
COPY ["predict_app.py", "model_train_py.bin", "./"]
#copy the files in the current directory into the container image

EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_app:app"]