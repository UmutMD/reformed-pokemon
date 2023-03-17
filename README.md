# Reformed-Pokemon

The first version is available at this link [Github-pokemon](https://github.com/UmutMD/pokemon-type-classification)

I converted my hobby project into a Docker file using the knowledge I have learned. In the first instance, the necessary libraries must be installed in the operating system or installed by the user before they can be used on another platform. This application, which has been converted into a container as a Dockerfile, will, however, operate on any operating system that can open a docker file.

## Installation and Usage

After downloading all of the files here, run the command below and wait for a while to see the outcomes.

```python
docker build -t reformed . 
docker run reformed
```

The DockerFile commands listed below will change the directory to where it should be when the container begins and provide the code that should be executed with CMD.

```Dockerfile

WORKDIR /reformed/src
CMD ["python3", "main.py"]
```
Start your Docker image with the commands below to see what's inside the container, and then execute the program again with the CMD code above.
```python
docker run --rm -it reformed bash
```
This code line will start a terminal in the container environment for you. 

## Procedure

After loading the required libraries, the images and csv file are loaded into our Python script. This material is divided into three categories: train, test, and validation. It then extracts features by building a 3-layer model with 'Adam' as the optimizer and 'binary cross entropy' as the loss. Finally, the weights of the pixel colors defined by the pokemon according to the model are displayed as a matrix on the screen.


## To Do Next

Run this image on AWS ECR server.

## Changes from 1.0 to 2.0

> Added documentation to the Project

> Pathing to local files fixed -- Fixed bugs in file read after converting to Dockerfile 

> Fixed requirement.txt libraries -- Some of them was unneccesary 

> Moved to ubuntu os image

> Created the DockerFile
