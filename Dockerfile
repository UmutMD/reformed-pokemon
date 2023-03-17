FROM ubuntu:latest
#FROM python: 30


# Set the working directory to /app
WORKDIR /reformed
#RUN apt update
#RUN apt install python3 -y
COPY requirements.txt .
COPY ./src ./src
RUN set -xe \
    && apt-get update -y\
    && apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.0-py3-none-any.whl
RUN pip install -r requirements.txt
RUN pip install scikit-learn
RUN pip install -U tensorflow
WORKDIR /reformed/src
#RUN apt-get install python3-pip
#RUN pip install --upgrade pip
# Get the 


#RUN pip install -r requirements.txt
#docker run -it -w /src reformed  
# docker run --rm -it reformed bash
CMD ["python3", "main.py"]

#docker rmi -f $(docker images -aq)