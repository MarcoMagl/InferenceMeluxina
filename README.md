# Meluxina Large Model Inference

This repository demonstrates how to run inference for a large-scale language model on **Meluxina**, Luxembourg's national supercomputer. The repository includes a SLURM batch script to set up and execute distributed inference of an HuggingFace model using Ray and vllm.&#x20;

## Prerequisites

To use this repository, you need access to Meluxina and:

1. **Hugging Face account**: Obtain an API token to access the model.
2. **Model access**: Ensure you are granted access to the Hugging Face model used in the script (in this example we need access to `mistralai/Mixtral-8x7B-Instruct-v0.1`).

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:MarcoMagl/InferenceMeluxina.git
cd InferenceMeluxina
```

### 2. Set Environment Variables

Before submitting the SLURM job, ensure the following environment variables are set:

- **HUGGINGFACEHUB\_API\_TOKEN**: Your Hugging Face API token.
  ```bash
  export HUGGINGFACEHUB_API_TOKEN=<your_token>
  ```
- **LOCAL\_HF\_CACHE**: Local directory for Hugging Face cache.
  ```bash
  export LOCAL_HF_CACHE=/path/to/local/cache
  ```

### 3. Check the Singularity Image File (SIF)

The script assumes the Singularity Image File (SIF) is named `vllm-openai_latest.sif` and located in the same directory as the script. If it’s not present, the script will automatically pull the image.

You can also manually pull the image:

```bash
apptainer pull vllm-openai_latest.sif docker://vllm/vllm-openai:latest
```

### 4.  SLURM Account and Configuration

Ensure the following SLURM configurations in `inference_meluxina.sh` match your account and requirements:

- **SLURM account** (`#SBATCH -A lxp`): **you must change this part** and replace it with your own account
- **Queue/partition** (`#SBATCH -p gpu`)
- **Node and resource allocation** (`#SBATCH -N 5`, `--gpus-per-task=4`, etc.)

### 5. Modify the Script for The Model you Want to Test

Update the model name in the script if needed:

```bash
export HF_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
```

Replace with your desired model name from Hugging Face.

## Running the Script

Once connected to Meluxina, submit the SLURM job using:

```bash
sbatch inference_meluxina.sh
```

### Output

- **Logs**: The output and error logs will be saved as `vllm-[job_id].out` and `vllm-[job_id].err` in the working directory.

### 6. Interacting with the Server

### SSH Tunnel From Your Local Machine

The script will provide instructions to set up an SSH tunnel for accessing the head node from your local machine. Simply open a terminal an copy and paste the command that you will find in the `vllm-[job_id].out` 
Be careful, every time that you will launch `sbatch inference_meluxina.sh`, it is likely that the IP of the head node will change and hence, you will have to copy and paste the up-to-date command in your terminal.

### Launching the chatbot

Open a new terminal on your machine. Do not close the one on which the ssh port forwarding is running. 
Copy and paste the `launch_chatbot.py` file to your local machine and run it with

```bash
python launch_chatbot.py
```

You can now reach the inference servir via the provided URL and interact with the model.

## Key Features

- **Distributed Inference**: The script uses Ray for distributed execution, enabling inference across multiple nodes and GPUs.
- **Containerized Execution**: Uses Apptainer to ensure a consistent runtime environment.
- **Parallelization**: Configurable tensor and pipeline parallelism for efficient model inference.

## Troubleshooting

1. **Missing `HUGGINGFACEHUB_API_TOKEN`**:

   - Ensure the API token is exported as an environment variable.
   - Use `export HUGGINGFACEHUB_API_TOKEN=<your_token>`.

2. **Singularity Image Not Found**:

   - Ensure the SIF image exists or let the script pull it automatically.

3. **Resource Allocation Errors**:

   - Adjust SLURM configurations (e.g., number of nodes, GPUs per task) to match your allocation.

4. **Model Access Denied**:

   - Confirm you have access to the specified Hugging Face model.

## License

This project is licensed under the MIT License.&#x20;

## Acknowledgments

- **LuxProvide** SAS Team
- **Hugging Face**: For their open-source models and APIs.
- **VLLM**: For their efficient model serving framework.

