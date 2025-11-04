.. _performanceconsiderations:
Performance Considerations
============================
This section discusses performance considerations when using gpuRDF2vec for training 
RDF2Vec models on large-scale knowledge graphs. The package offers two different modes of operation:

1. **Single-GPU Mode**: In this mode, the entire training process is executed on a single GPU.
This is suitable for small to medium-sized knowledge graphs that fit into the memory of a single 
GPU.

2. **Multi-GPU Mode**: This mode leverages multiple GPUs for distributed training of RDF2Vec 
embeddings. This is particularly useful for large knowledge graphs that exceed the memory capacity of a single GPU.
When using multi-GPU mode, the package utilizes PyTorch's Distributed Data Parallel (DDP) 
framework along with Dask-CUDA to distribute the workload across multiple GPUs. 
This allows for efficient training of embeddings on large graphs by splitting the data and 
computation across the available GPUs.

Single-GPU Mode
-----------------
In single-GPU mode, the entire RDF2Vec training process is performed on a single GPU. 
This mode is straightforward to set up and is suitable for GB scale graphs with around 
50 mio edges within the graph and is able to run.

While not directly documented within the cudf package, there is a general limitation how big a 
cudf dataframe can be. This is mainly due to the fact that cudf dataframes use 32-bit indexing, 
which limits the maximum size of a single cudf dataframe to 2^31 - 1 rows. Therefore, when working 
with very large graphs, it is important to ensure that the size of the data internally processed in the library 
does not exceed this limit.
A discussion on this topic can be found in the following `GitHub issue cudf#6216 <https://github.com/rapidsai/cudf/issues/7991#issuecomment-822417240>`_. In such a case,
it is recommended to use the multi-GPU mode for training RDF2Vec embeddings.

The single-GPU mode is enabled by setting the `multi_gpu` parameter to `False` when initializing
the `GPU_RDF2Vec` class.

General performance recommendations for single-GPU mode:

- Ensure that the graph data fits into the memory of the GPU.
- Monitor GPU memory usage during training to avoid out-of-memory errors.
- Use appropriate batch sizes to optimize GPU utilization

Multi-GPU Mode
-----------------
In multi-GPU mode, the training process is distributed across multiple GPUs using PyTorch's
Distributed Data Parallel (DDP) framework along with Dask-CUDA. This allows for handling
large-scale knowledge graphs that exceed the memory capacity of a single GPU.
To enable multi-GPU mode, set the `multi_gpu` parameter to `True` when initializing the `GPU_RDF2Vec` class.
When using multi-GPU mode, it is important to set up a Dask-CUDA cluster to manage the distribution of tasks 
across the GPUs.

This can generally include many different configurations, ranging from single-node multi-GPU 
setups to multi-node multi-GPU clusters. Detailed instructions on setting up a Dask-CUDA 
cluster can be found under the `distributed training <advanced/distributedtraining>`_ section of the documentation.

In it's current design the multi-GPU mode is mainly targeted towards large-scale knowledge graphs with
hundreds of millions of edges and beyond. The performance benefits of using multi-GPU mode will
depend on the specific graph size, the number of available GPUs, and the configuration of the 
Dask-CUDA cluster. Another big influence factor is the way how the data is partitioned across the different
workers and GPUs. It is recommended to experiment with different configurations to find the optimal setup
for your specific use case.

**Currently open**: We offer a relatively high abstraction on the underlying interfaces when a user
interacts with the library. In future releases we plan to provide more fine granular interfaces
to allow users to customize the different steps based on the specific use case. This allows to 
persist data between the steps and generally benefit the multi-GPU support and distributed training
capabilities in order to reduce the task graph of Dask for very large graphs.

General performance recommendations for multi-GPU mode:

- Set up a Dask-CUDA cluster that matches the available hardware resources.
- Ensure that the graph data is partitioned equally across the GPUs (dask worker). Unbalanced data distribution can lead to suboptimal performance.
- Experiment with different data partitioning strategies to optimize workload distribution.
- Monitor the performance of each GPU to ensure balanced utilization across all devices using the dask dashboard.