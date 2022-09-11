Team name:
联邦对不队

1. Preparing the federatedscope environment

Follow the official tutorial to build the federatedscope environment.

Our team's code execution environment is Centos, GPU is V100 with 32G RAM.

2. To run the code:

> python federatedscope/main.py --cfg federatedscope/gfl/baseline/final_a.yaml --client_cfg federatedscope/gfl/baseline/final.yaml

3. Brief introduction of the developed algorithm:

All clients are trained based on the GIN model by isolated training (no federated learning methods are used)

GIN refer from Graph Isomorphism Network model from the "How Powerful are Graph
    Neural Networks?" paper, in ICLR'19
