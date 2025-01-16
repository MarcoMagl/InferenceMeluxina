#!/bin/bash -l
# :-) do not forget the -l after the shebang or your script won't work on Meluxina!

# Authors:
# - Marco Magliulo
# - Emmanuel Kieffer
# - Wahid Mainassara
#
# Affiliation:
# LuxProvide - Luxembourg National Supercomputer

# This script launches a distributed training or inference job using vllm.
# It configures environment variables, checks prerequisites, and coordinates a head node and worker nodes.

#SBATCH -A lxp                     # Specify the SLURM account to use. IF YOU WANT TO RUN THIS SCRIPT CHANGE THIS LINE USING YOUR PROJECT ID LIKE p200000
#SBATCH -q default                 # Specify the queue (partition) to submit the job to. Default is fine here 
#SBATCH -p gpu                     # Specify that the job requires GPU resources.
#SBATCH -t 2:0:0                   # Set a 2-hour time limit for the job.
#SBATCH -N 5                       # Request 5 nodes (1 head node + 4 worker nodes).
#SBATCH --ntasks-per-node=1        # Use 1 task per node.
#SBATCH --cpus-per-task=128        # Allocate 128 logical cores for each task.
#SBATCH --gpus-per-task=4          # Allocate 4 GPUs for each task --> On Meluxina there are 4GPUs per node, we want to use them all
#SBATCH --error="vllm-%j.err"      # Redirect standard error to a file named vllm-[job_id].err.
#SBATCH --output="vllm-%j.out"     # Redirect standard output to a file named vllm-[job_id].out.
#SBATCH --reservation=gpu-llm-inference-meluxina #This is the reservation I will use for the 2025/01/17 talk. If you want to run this script, remove this line !! 

# Load necessary modules
module --force purge               # Purge all currently loaded modules to ensure a clean environment.
module load env/release/2023.1     # Load the environment release module. Here we use the 2023 software stack, but we could have used the 24 release stack too 
module load Apptainer/1.3.1-GCCcore-12.3.0  # Load Apptainer (formerly Singularity) for containerized execution.

# Export the Hugging Face API token from the environment variable
export HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN}

# Check if the Hugging Face token is set, and exit if it's not. We won't be able to get the model shards without that 
if [ -z "${HUGGINGFACEHUB_API_TOKEN}" ]; then
    echo "Warning: HUGGINGFACEHUB_API_TOKEN is not defined. Please set it in this script or in your bashrc before proceeding."
    exit 1
fi

# Set PMIX security mechanism to native to fix potential PMIX errors
export PMIX_MCA_psec=native

# Check if the LOCAL_HF_CACHE directory is set
# BE CAREFUL: HF will download the model shards in this directory. A lot of space might be needed 
# On Meluxina, use the myquota command from the terminal to see where you have enough space to download the shards
if [ -z "${LOCAL_HF_CACHE}" ]; then
    echo "Warning: LOCAL_HF_CACHE is not defined. Please set it in this script or in your bashrc before proceeding."
    exit 1
fi

# Print and ensure the Hugging Face cache directory exists
echo "LOCAL_HF_CACHE is set to: $LOCAL_HF_CACHE"
mkdir -p ${LOCAL_HF_CACHE}

# Define the Singularity Image File (SIF) we will use 
# We want to use the vllm image that contains to make a long story short, everything we need to run the inference model
export SIF_IMAGE="vllm-openai_latest.sif"

# Check if the SIF image exists, and pull it if not
if [ ! -f "$SIF_IMAGE" ]; then
    echo "File $SIF_IMAGE not found. Pulling the image..."
    apptainer pull "$SIF_IMAGEFILE" docker://vllm/vllm-openai:latest
else
    echo "File $SIF_IMAGE already exists. Skipping pull."
fi

# Configure Apptainer arguments for container execution
#    --nvccli: Ensures NVIDIA GPU resources are available inside the container. Apptainer injects the appropriate NVIDIA runtime dependencies and libraries into the container.
#    -B: Maps the host’s cache directory to the container’s Hugging Face cache directory for persistence.

export APPTAINER_ARGS=" --nvccli -B ${LOCAL_HF_CACHE}:/root/.cache/huggingface\
        --env HF_HOME=/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"

# Define the Hugging Face model to use (ensure you have access to it)
export HF_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"

# Set the head node's hostname and IP address
# This is the first node of the pool of selected nodes by default!
export HEAD_HOSTNAME="$(hostname)"
export HEAD_IPADDRESS="$(hostname --ip-address)"

# Print the head node details and SSH tunneling instructions
echo "HEAD NODE: ${HEAD_HOSTNAME}"
echo "IP ADDRESS: ${HEAD_IPADDRESS}"

#IMPORTANT: this is the command that you will have to copy and paste in the terminal of your local machine!
echo "SSH TUNNEL (Execute on your local machine): ssh -p 8822 ${USER}@login.lxp.lu  -NL 8000:${HEAD_IPADDRESS}:8000"

# Get a random available port for Ray
export RANDOM_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Define commands to start Ray on the head node and worker nodes
# The --block flag keeps the worker process running in the foreground.
# Without --block, the Ray worker process would start and then immediately exit, which is not desirable for a persistent worker node.
export RAY_CMD_HEAD="ray start --block --head --port=${RANDOM_PORT}"
# The --address specifies the address of the head node to which this worker node should connect.
export RAY_CMD_WORKER="ray start --block --address=${HEAD_IPADDRESS}:${RANDOM_PORT}"

# Set parallelization configurations for the model
export TENSOR_PARALLEL_SIZE=4                 # Number of GPUs per node for tensor parallelism.
export PIPELINE_PARALLEL_SIZE=${SLURM_NNODES} # Number of nodes for pipeline parallelism.

# Start the head node using SLURM and Apptainer
echo "Starting head node"
srun -J "head ray node-step-%J" -N 1 --ntasks-per-node=1 -c $((SLURM_CPUS_PER_TASK / 2)) -w ${HEAD_HOSTNAME} apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} ${RAY_CMD_HEAD} &
sleep 10  # Wait for the head node to initialize

# Start the worker nodes using SLURM and Apptainer
echo "Starting worker node"
srun -J "worker ray node-step-%J" -N $((SLURM_NNODES - 1)) --ntasks-per-node=1 -c ${SLURM_CPUS_PER_TASK} -x ${HEAD_HOSTNAME} apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} ${RAY_CMD_WORKER} &
sleep 10  # Wait for the worker nodes to initialize

# Start the model server on the head node
echo "Starting server"
apptainer exec ${APPTAINER_ARGS} ${SIF_IMAGE} vllm serve ${HF_MODEL} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE}

