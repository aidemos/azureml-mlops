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
            print('found compute target. just use it. ' + compute_name)
            return compute_target
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

def run_pipeline(
    ws: Workspace,
    pipeline_id: str,
    experiment_name:str,
    ):

    experiment = Experiment(ws, experiment_name)
    published_pipeline = PublishedPipeline.get(workspace=ws, id=pipelineid)
    pipeline_run = experiment.submit(published_pipeline)
    print("pipeline run submitted")