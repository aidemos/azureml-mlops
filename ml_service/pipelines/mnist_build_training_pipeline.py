import os
import tarfile
import urllib.request
from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.datastore import Datastore
from azureml.core.dataset import Dataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.model import Model
from azureml.core import Environment
from azureml.core.runconfig import CondaDependencies, DEFAULT_CPU_IMAGE
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.core import PublishedPipeline
from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig, PythonScriptStep
from azureml.core import Experiment
from ml_service.util.env_variables import Env
from ml_service.util.aml_helper import get_workspace, get_compute, run_pipeline
from azureml.opendatasets import MNIST
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Environment 

def get_training_dataset(
    ws: Workspace
):
    from azureml.core import Dataset
    from azureml.opendatasets import MNIST

    data_folder = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_folder, exist_ok=True)

    mnist_file_dataset = MNIST.get_file_dataset()
    mnist_file_dataset.download(data_folder, overwrite=True)

    mnist_file_dataset = mnist_file_dataset.register(workspace=ws,
                                                    name='mnist_opendataset',
                                                    description='training and test dataset',
                                                    create_new_version=True)

    return mnist_file_dataset

def get_training_run_config(
    compute_target: ComputeTarget,
):
    aml_run_config = RunConfiguration()
    # `compute_target` as defined in "Azure Machine Learning compute" section above
    aml_run_config.target = compute_target
    aml_run_config.environment.python.user_managed_dependencies = False
    
    # Add some packages relied on by data prep step
    aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=['azureml-dataset-runtime[pandas,fuse]', 'azureml-defaults'], conda_packages = ['scikit-learn==0.22.1'])
    return aml_run_config

def get_training_pipeline(
    ws: Workspace,
    compute_target: ComputeTarget,
    input_mnist_ds_consumption: Dataset,
    training_run_config = RunConfiguration,
):
    scripts_folder = "mnist/training"
    script_file = "train.py"

    training_step = PythonScriptStep(
    source_directory = scripts_folder,
    script_name= script_file,
    arguments=['--data-folder', input_mnist_ds_consumption.as_mount(), '--regularization', 0.5],
    name="mnist-training",
    compute_target=compute_target,
    runconfig=training_run_config
    )

    pipeline = Pipeline(workspace=ws, steps=[training_step])

    return pipeline

def build_training_pipeline():
    """
    Main method that builds and publishes a scoring pipeline.
    """
    try:
        env = Env()
        ws = get_workspace(env)
        compute_target = get_compute(ws)
        input_dataset = get_training_dataset(ws)
        training_run_config = get_training_run_config(compute_target)
        pipeline = get_training_pipeline(ws,compute_target,input_dataset,training_run_config)
        published_pipeline = pipeline.publish(
            name="mnist_training",#env.scoring_pipeline_name,
            description="MNIST Training Pipeline",
        )
        pipeline_id_string = published_pipeline.id     
        print(pipeline_id_string)
        return pipeline_id_string
    except Exception as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    env = Env()
    ws = get_workspace(env)
    pipeline_id = build_training_pipeline()
    run_pipeline(ws,pipeline_id,"mnist_training")
