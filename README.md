## Algorithms for reasoning with Conditional Independence Graphs

Conditional Independence (CI) graph is a special type of Probabilistic Graphical Model (PGM) where the feature connections are modeled using an undirected graph and the edge weights show the partial correlation strength between the features. Since, the CI graphs capture direct dependence between features, they have been widely used for gaining insights into the systems from various domains. These graphs have primarily been used to discover feature connections and its topology. In this repository, we focus on algorithms for reasoning with these CI graphs. Particularly, we developed algorithms for doing knowledge propagation, graph compression, data imputation and probabilistic reasoning that allow us to perform inference using the distribution represented by the graph.  

## Setup  
The `setup.sh` file contains the complete procedure of creating a conda environment to run mGLAD model. run `bash setup.sh`    
In case of dependencies conflict, one can alternatively use this command `conda env create --name rCI --file=environment.yml`.  

## Demo notebook    
Minimalist toy examples are given in `demo.ipynb`. It is a good entry point to understand the code structure.   

## Citation  
If you find our work useful, kindly cite the following associated papers:
- `Algorithms for reasoning with Conditional Independence Graphs`: [arxiv](<>)

- `Methods for Recovering Conditional Independence Graphs: A Survey`: [arxiv](<https://arxiv.org/abs/2211.06829>)  
@article{shrivastava2022methods,  
  title={Methods for Recovering Conditional Independence Graphs: A Survey},  
  author={Shrivastava, Harsh and Chajewska, Urszula},  
  journal={arXiv preprint arXiv:2211.06829},  
  year={2022}  
}  