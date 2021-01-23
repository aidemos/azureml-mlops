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
from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig
from azureml.core import Experiment
from ml_service.util.env_variables import Env
from ml_service.util.aml_helper import get_workspace, get_compute, get_file_dataset, run_pipeline

def get_output_dir(
    ws: Workspace
):
    def_data_store = ws.get_default_datastore()
    output_dir = PipelineData(name="inferences", datastore=def_data_store)
    return output_dir

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

def get_batch_environment():
    batch_conda_deps = CondaDependencies.create(pip_packages=["tensorflow==1.15.2", "pillow", 
                                                            "azureml-core", "azureml-dataset-runtime[fuse]"])
    batch_env = Environment(name="batch_environment")
    batch_env.python.conda_dependencies = batch_conda_deps
    batch_env.docker.enabled = True
    batch_env.docker.base_image = DEFAULT_CPU_IMAGE
    return batch_env

def get_scoring_pipeline(
    ws: Workspace,
    batch_env: Environment,
    compute_target: ComputeTarget,
    input_mnist_ds_consumption: Dataset,
    output_dir: PipelineData
):
    scripts_folder = "mnist"
    script_file = "inferencing/digit_identification.py"
    # Define parallel run config
    parallel_run_config = ParallelRunConfig(
        source_directory=scripts_folder,
        entry_script=script_file,
        mini_batch_size=PipelineParameter(name="batch_size_param", default_value="5"),
        error_threshold=10,
        output_action="append_row",
        append_row_file_name="mnist_outputs.txt",
        environment=batch_env,
        compute_target=compute_target,
        process_count_per_node=PipelineParameter(name="process_count_param", default_value=2),
        node_count=2
    )

    # ### Create the pipeline step
    # Create the pipeline step using the script, environment configuration, and parameters. Specify the compute target you already attached to your workspace as the target of execution of the script. We will use ParallelRunStep to create the pipeline step.
    parallelrun_step = ParallelRunStep(
        name="predict-digits-mnist",
        parallel_run_config=parallel_run_config,
        inputs=[ input_mnist_ds_consumption ],
        output=output_dir,
        allow_reuse=False
    )

    pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])

    return pipeline

def build_batchscore_pipeline():
    """
    Main method that builds and publishes a scoring pipeline.
    """
    try:
        env = Env()
        ws = get_workspace(env)
        compute_target = get_compute(ws)
        input_dataset = get_file_dataset(ws)
        register_model(ws)
        output_dir = get_output_dir(ws)
        batch_env = get_batch_environment()
        pipeline = get_scoring_pipeline(ws,batch_env,compute_target,input_dataset,output_dir)
        published_pipeline = pipeline.publish(
            name="image_inferencing",#env.scoring_pipeline_name,
            description="MNIST Batch Scoring Pipeline",
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
    pipeline_id = build_batchscore_pipeline()
    run_pipeline(ws,pipeline_id,'mnist_inferencing')
