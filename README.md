# MLOps with Azure ML

This repo shows a simple way of developing Azure ML pipelines via python code and then deploying them via Azure DevOps. This is an excerpt from the [mlops-python repo](https://github.com/microsoft/MLOpsPython) and relies on the instructions for this repo until [this step](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#set-up-build-release-trigger-and-release-multi-stage-pipelines)

The files in the folder .pipelines are Azure DevOps pipelines and the files in the folder ml_service are Azure Machine Learning pipelines along with some utilities. 

The files in the folder endpoint contain files required for AKS deployment.

The mnist folder contains python files for training and inferencing pipelines. 

To Do:
* Include a separate repo for Infrastructure as Code
* Include automated creation of the Azure DevOps project and variables
