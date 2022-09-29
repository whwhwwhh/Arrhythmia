FROM python:3.7

USER root

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY . /app
RUN make /app
CMD python /app/job/ml_job.py