# pytorch-cifar-benchmarking

Implementing and Benchmarking ResNet-18 in Pytorch on the CIFAR10 dataset. 
Model Architecture c/o [He, Zhang et. al](https://arxiv.org/abs/1512.03385)

For use with Columbia University's High Performance Computing Cluster, [Habanero](https://confluence.columbia.edu/confluence/display/rcs/Habanero+-+Getting+Started)

## Usage

~~~
load module conda
   conda create -name hw2
   conda activate hw2
   conda install pytorch
   conda install torchvision
   conda install prettytables 
   
   
   ~~~
   
Followed by

   ``` sbatch execute.sh```
