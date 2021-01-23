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
from ml_service.util.aml_helper import get_workspace
from ml_service.util.aml_helper import get_compute
from azureml.opendatasets import MNIST
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Environment 

# def get_file_dataset(
#     ws: Workspace
# ):

#     data_folder = os.path.join(os.getcwd(), 'data')
#     os.makedirs(data_folder, exist_ok=True)

#     mnist_file_dataset = MNIST.get_file_dataset()
#     mnist_file_dataset.download(data_folder, overwrite=True)

#     mnist_file_dataset = mnist_file_dataset.register(workspace=ws,
#                                                     name='mnist_opendataset',
#                                                     description='training and test dataset',
#                                                     create_new_version=True)
#     return mnist_file_dataset

def get_file_dataset(
    ws: Workspace
):
    # ### Create a datastore containing sample images
    account_name = "pipelinedata"
    datastore_name = "mnist_datastore"
    container_name = "sampledata"

    mnist_data = Datastore.register_azure_blob_container(ws, 
                        datastore_name=datastore_name, 
                        container_name=container_name, 
                        account_name=account_name,
                        overwrite=True)

    def_data_store = ws.get_default_datastore()

    mnist_ds_name = 'mnist_sample_data'

    path_on_datastore = mnist_data.path('mnist')
    input_mnist_ds = Dataset.File.from_files(path=path_on_datastore, validate=False)

    pipeline_param = PipelineParameter(name="mnist_param", default_value=input_mnist_ds)
    input_mnist_ds_consumption = DatasetConsumptionConfig("minist_param_config", pipeline_param).as_mount()

    return input_mnist_ds_consumption

def register_model(
    ws: Workspace
):
    # ### Download the Model
    # create directory for model
    model_dir = 'models'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    url="https://pipelinedata.blob.core.windows.net/mnist-model/mnist-tf.tar.gz"
    response = urllib.request.urlretrieve(url, "model.tar.gz")
    tar = tarfile.open("model.tar.gz", "r:gz")
    tar.extractall(model_dir)

    os.listdir(model_dir)

    # ### Register the model with Workspace
    model = Model.register(model_path="models/",
                        model_name="mnist", # this is the name the model is registered as
                        tags={'pretrained': "mnist"},
                        description="Mnist trained tensorflow model",
                        workspace=ws)
    print("model registered")

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
    name="create-dataset",
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
        input_dataset = get_file_dataset(ws)
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

def run_pipeline(
    ws: Workspace,
    pipelineid: str,
    ):

    experiment = Experiment(ws, 'mnist_training')
    published_pipeline = PublishedPipeline.get(workspace=ws, id=pipelineid)
    pipeline_run = experiment.submit(published_pipeline)
    print("pipeline run submitted")

if __name__ == "__main__":
    env = Env()
    ws = get_workspace(env)
    pipeline_id = build_training_pipeline()
    run_pipeline(ws,pipeline_id)
