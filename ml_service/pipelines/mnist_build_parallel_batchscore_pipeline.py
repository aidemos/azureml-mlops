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
from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig
from azureml.core import Experiment
from ml_service.util.env_variables import Env

def get_workspace(env: Env):
        #Get Azure machine learning workspace
        ws = Workspace.get(
            name=env.workspace_name,
            subscription_id=env.subscription_id,
            resource_group=env.resource_group,
        )

        print("got workspace")

        return ws

def get_compute(
    ws: Workspace
):
    # choose a name for your cluster
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpu-cluster")
    compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
    compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

    # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")


    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            return compute_target
            print('found compute target. just use it. ' + compute_name)
    else:
        print('creating a new compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                    min_nodes = compute_min_nodes, 
                                                                    max_nodes = compute_max_nodes)

        # create the cluster
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
        
        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
        
        # For a more detailed view of current AmlCompute status, use get_status()
        print(compute_target.get_status().serialize())
        return compute_target

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
    return build_batchscore_pipeline()
