.. _distributedtraining:

Distributed Training with gpuRDF2vec
======================================
The `gpuRDF2vec` package supports distributed training of RDF2vec embeddings across multiple GPUs
using PyTorch's Distributed Data Parallel (DDP) framework together with the distributed Rapids engines
with the combination of Dask. This allows for operating graphs that do not fit directly into a single
GPU's memory and speeds up the training process by leveraging multiple GPUs for multiple workers in 
parallel.

Setting up Local Cuda cluster (Single node Multi-GPU paradigm)
--------------------------------------------------------------
To utilize distributed training with `gpuRDF2vec`, you need to set up a CUDA cluster using Dask-CUDA.
Here is a basic example of how to set up a Dask-CUDA cluster. A detailed guide on setting up a local CUDA
cluster can be found in the `Dask-CUDA documentation <https://docs.rapids.ai/api/dask-cuda/stable/api/>`_

.. code-block:: python

    from dask_cuda import LocalCUDACluster

    from dask.distributed import Client

    # Create a local CUDA cluster
    cluster = LocalCUDACluster()
    
    client = Client(cluster)

Setting up Multi-node Multi-GPU cluster with SLURM
-------------------------------------------------
To set up a multi-node multi-GPU cluster using SLURM, you can use the `dask-jobqueue` package 
to create a SLURM cluster which specifies the number of GPUs per node and other parameters 
required for the starting of the SLURM job. 

**Important**: When using SLURM, ensure that all the dependent packages (like RAPIDS, Dask, PyTorch, etc.)
are installed and accessible on all nodes in the cluster.

An example of setting up a SLURM cluster is shown below:

.. code-block:: python

    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client

    cluster = SLURMCluster(
        queue='gpu_partition',  # Specify your SLURM GPU partition
        cores=1,                # Number of cores per worker
        memory='32GB',          # Memory per worker
        walltime='01:00:00',    # Walltime for the SLURM job
        worker_extra_args=['--gres=gpu:1'], # Request 1 GPU per job
        # Add any other SLURM-specific options like account, etc.
    )
    cluster.scale(jobs=4)
    client = Client(cluster)

Setting up Multi-node Multi-GPU cluster over cloud environments (**not tested**)
--------------------------------------------------------------------------------
Setting up a multi-node multi-GPU cluster over cloud environments (like AWS, GCP, Azure) generally involves using managed services or setting up virtual machines
with GPU capabilities. You can use Dask's distributed scheduler to connect to these nodes.
A description of setting up Dask clusters on various cloud platforms can be found in the
following `rapids documentation <https://docs.rapids.ai/deployment/stable/cloud/>`_.

Using cluster with gpuRDF2vec
------------------------------
Once you have set up your Dask-CUDA cluster (either local or SLURM-based), you can pass the Dask 
client to the class instance of `GPU_RDF2Vec` when initializing it. This will enable the 
distributed training capabilities of the package.

Here is an example of how to use the Dask client with GPU_RDF2Vec for distributed training.

.. code-block:: python

    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    
    # Set up a local CUDA cluster
    cluster = LocalCUDACluster()
    client = Client(cluster)
    
    # Initialize the GPU_RDF2Vec model with the Dask client for distributed training
    gpu_rdf2vec_model = GPU_RDF2Vec(
          walk_strategy="random",
          walk_depth=4,
          walk_number=100,
          embedding_model="skipgram",
          epochs=5,
          batch_size=None,
          vector_size=100,
          window_size=5,
          min_count=1,
          learning_rate=0.01,
          negative_samples=5,
          random_state=42,
          reproducible=False,
          multi_gpu=True,  # Enable multi-GPU training
          client=client  # Pass the Dask client here
     )
    
    path = "data/wikidata5m/wikidata5m_kg.parquet"
    # Load data and receive edge data
    edge_data = gpu_rdf2vec_model.load_data(path)
    # Fit the Word2Vec model and transform the dataset to an embedding
    embeddings = gpu_rdf2vec_model.fit_transform(edge_df=edge_data, walk_vertices=None)