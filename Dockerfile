FROM python:3
RUN pip install --upgrade pip
RUN pip install jupyterlab
RUN pip install -U "jupyter-server<2.0.0"
# Set the working directory to /app
WORKDIR /uuh

# Get the 
COPY requirements.txt .
COPY ./src ./src

RUN pip install -r requirements.txt

# Configuring access to Jupyter
RUN mkdir /notebooks
RUN jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 /notebooks --allow-root
