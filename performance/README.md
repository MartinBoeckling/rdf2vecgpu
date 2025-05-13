# Performance comparison setup
This folder contains everything you need to build and run the performance benchmarks for the GPU-accelerated RDF2Vec implementation. Use these benchmarks to measure throughput, memory usage, and speedup over the CPU baseline.

## Dataset creation
Please download the set of synthetic datasets from the following [Zenodo repository](https://doi.org/10.5281/zenodo.15368485) and place them under `data/generated_graphs`. Please use the FB15k-237 repository as well as the wikidata-5m repository in order to download the data assets. For both the FB15k-237 dataset as well as the [merge_text_file script](data_preparation/merge_text_file.py) to merge all the different splits together. 

For both the jrdf2vec library as well as the sparkkgml library please convert the parquet files into a ttl file.

Scripts:
- To calculate the graph statistics please run the following [script](graph_statistics.py)
- To run the jrdf2vec performance comparison, please use the following [script](jrdf2vec_based_performance.py)
- To run the pyrdf2vec performance comparison, please use the following [script](pyrdf2vec_based_performance.py)
- To run the sparkrdf2vec performance comparison, please use the following [script](spark_rdf2vec_performance.py)
- To run the gpu_rdf2vec performance comparison, please use the following [script](gpu_rdf2vec_performance.py)