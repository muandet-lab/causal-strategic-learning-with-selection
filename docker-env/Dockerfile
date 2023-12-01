FROM --platform=linux/amd64 python:3.8-buster

RUN mkdir /home/guest
WORKDIR /home/guest

COPY requirements.txt ./
RUN pip install -r requirements.txt

# To be removed
RUN pip install notebook

# To be run with bind mounts

ENV PYTHONPATH=.
