# example launch script

# common arguments for all methods
# BASE_ARGS="--num_clients 100 --sampling_prob 0.3 --local_epochs 5 --global_rounds 200 --eval_gap 20"
BASE_ARGS="--num_clients 100 --sampling_prob 0.3 --local_epochs 5 --global_rounds 200 --eval_gap 20"
# data-scarcity arguments: e.g., train prop [0.25, 0.5, 0.75, 1.0]
# if we have created augmented datasets, we can also include --augmented to include covariate shift
DATA_ARGS="--train_prop 1.0"

# select model e.g., ['cnn', 'resnet18']
MODEL_ARGS="--model_name cnn"

# method-specific arguments
FEDAVG_ARGS="--method FedAvg"
FEDFDA_ARGS="--method pFedFDA"

# specify dataset arguments
DATASET_ARGS="--dataset cifar100 --num_classes 100"

# specify dataset partition arguments
PARTITION_ARGS="--partition_path cifar100_c100_dir05_25"

# FedAvg | FedAvgFT
nohup python main.py ${BASE_ARGS} ${FEDAVG_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS} >> FedAvg_multi_c100_cifar100_25_flow.log

# pFedFDA
# nohup python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS} >> pFedFDA_multi_c100_cifar100_25_flow.log
# python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS}
# pFedMDG
# nohup python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS} >> 1.log
# python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS}