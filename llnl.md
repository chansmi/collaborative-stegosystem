Using LC (Livermore Computing) resources
Tech support
Call LC support using the following number, they’re generally quick to help/respond. (925) 422-4531.

For more general IT questions (e.g. password/email/etc.), contact LivIT instead at 925-424-4357.
Storage
20TB Personal storage: 
/p/vast1/$USER/p
Good for training datasets, model states, etc.
2TB Personal storage: 
/usr/workspace/$USER/
Good for git repositories, conda envs, etc.
40TB Group storage for UMD collaborators: 
/p/vast1/pretrain/
Good for training datasets, model states, etc. that you want to share in a common space

Globus
Follow instructions here to request access to the service: https://hpc.llnl.gov/services/green-collaboration-environment/globus 
Then, large files can be transferred between OLCF, UMD (cml), and LLNL. 
On the LLNL side, the data transfer nodes are `goblin*` and the main attached storage is /p/globfs/$USER which is large, and has a “good citizen” quota/moderation policy (currently).
The data transfer nodes cannot see `lustre*` nor `/p/vast1` so, a second step to get the data from the globfs to the compute attached filesystems is required.
Note/Warning: the second step tool required for this is a simpler tool than globus, based on FTP. It works best with single large files (like tar.gz archives) so if possible, compress any dirs into such archives before all of the data movement ops and inflate them in the desired, compute accessible storage. There are some suggestions for how to transfer dir structures, but jwk hasn’t made this work yet (4/15/25).
 
My simplified version of the globfs -> compute fs’ transfer workflow is:
log onto `oslic` (oslic can see both globfs and the compute fs’)
invoke the `globftp` tool (and log in with uname then pwd)
`cd` to the src directory under `/p/globfs` where the files are that you want to move from globfs into the compute fs
`lcd` to the desired dest directory on say `/p/vast1` or `/p/lustre*` within the compute fs where you want files to land
then run `get source_filepath` and this will transfer the file at `source_filepath` on `/p/globfs` to the local directory (i.e. `lcd` argument in the prev step). You will see some occasional progress printing for large transfers. tmux is recommended if it’s big.



Bash/bashrc issues and notes
If your bashrc isn’t running automatically on login, please add to your home directory a file called `.bash_profile` with the following three lines:
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi
Note: If your bashrc sets up a particular conda installation, you can ensure this setup is only run on login nodes (lassen708-lassen710) by putting the conda setup code in an if-then block, see the following example:
if [ "$(hostname)" == "lassen710" ] || [ "$(hostname)" == "lassen709" ] || [ "$(hostname)" == "lassen708" ]; then
  # Load conda (note: does not activate env):
  export CONDAPATH=/usr/workspace/$USER/power9_miniconda
  eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
fi
Why this matters: Typically, you will activate a conda environment on a login node, submit a job, and the job will run on a job node using the environment you had activated on the login node when the job was submitted. Without the above if-statement, the environment on the job node (e.g., lassen750) will be reset to the “base” environment when `.bash_profile` calls `.bashrc` when the job scheduler logs you into the job node. TLDR: the above if-statement will ensure that your conda environment on the login (i.e., job-submission) node is also used to run the job on the job node.

You might want to also stick the following module loading command into the above if-block: `ml gcc/8.3.1 cuda/11.8.0 cmake/3.23.1 git/2.31.1`.

When using VIM, if backspaces just create “^?” symbols: make sure your .bashrc has the line 
stty erase '^?'
in it, then this issue should go away when the .bashrc file is sourced.
Jupyter
https://lc.llnl.gov/jupyter/hub/

Once logged in, replace “tree” in the URL with “lab” for jupyter lab.

To use your LC python environments in this jupyter tool, use a terminal to go to your home directory, activate the environment, and run `python -m ipykernel install --user`.

Tuolumne/Tuo (x86, MI300A)
Each tuo/tuolumne node has 4 GPUs.

This machine is like tioga, it uses flux. Helpful commands to see who is running things and what resources are available:
flux jobs -u all
flux resource list 
Logging in
ssh username@tuo.llnl.gov (does not require connecting to the VPN)
ssh username@tuo (requires connecting to the VPN)
Environment/Anaconda setup
The setup process for tioga should also work on Tuo. I’d recommend installing a bleeding edge pytorch, as this might get you optimal compatibility with AMD

When installing/using torch, you probably want to first load the appropriate ROCm module (based on the torch version you’re installing). E.g., call `ml rocm/6.3`.

ROCm-6.4 should be out in April 2025 and will supposedly bring a large 16-bit training speed boost.
Running jobs
Running jobs on tuo is the same as it is for tioga. 

An example launch command for litgpt style training on multiple nodes is here: https://github.com/bbartoldson/AMD-LLM/blob/main/rccl_launch.sh. Note that this example has way too many flags set and reset – i was trying various things to get better performance and wasn’t very careful about tidiness. Hopefully, a single best way to set flags and launch jobs emerges soon.

John Kirchenbauer and Sean are working on a new launching automation to explore toggle space a bit more systematically.
https://github.com/tomg-group-umd/llnl-tools 
VSCode+Copilot
If you run into an issue where you want to use copilot inside of vscode, but when remoted to tioga/tuo, you see a weird error where its not offering completions/working properly, and in the status bar you see a complaint about certificates and whatnot, try installing this extension
https://marketplace.visualstudio.com/items?itemName=linhmtran168.mac-ca-vscode 
Lassen (Power9, V100)
Each lassen node has 4 GPUs.
Logging in
ssh username@lassen.llnl.gov (does not require connecting to the VPN)
ssh username@lassen (requires connecting to the VPN)
Anaconda setup
Automatic approach
This is basically automatic, despite the steps. You will need to paste some code into some files, then run the files and possibly hit enter / say yes to some prompts. 
Step 1
I wrote a shell script that completes for you the steps listed below under “Manual approach”. Paste the following into a file named “power_conda_setup.sh” then run `sh power_conda_setup.sh`:
```
cd /usr/workspace/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
# load some basic dependencies that may be helpful
ml gcc/8.3.1 cuda/11.8.0 cmake/3.23.1 git/2.31.1
sh Miniconda3-latest-Linux-ppc64le.sh -b -p power_miniconda/
export CONDAPATH=/usr/workspace/$USER/power_miniconda
eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
conda config --append channels https://ftp.osuosl.org/pub/open-ce/current/
conda config --append channels https://opence.mit.edu
conda config --append channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
conda config --append channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access
conda config --append channels defaults
conda update conda
conda install  --force-reinstall openssl=1.1.1
conda config --set ssl_verify /etc/pki/tls/cert.pem
```
# Installing pytorch in a new env would involve these extra commands:
```
conda create -y -n env_name
conda activate env_name
ml cuda/11.8.0 gcc/11.2.1 
conda install -y  pytorch=2.0.1 cudatoolkit=11.8.0
```
Step 2
Add a file to your home directory called “.bash_profile” with the following three lines:
```
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi
```
Step 3
Create an environment. E.g., create the mega_axonn environment by running `source power_megatron_axonn_setup.sh`, where the shell script’s contents are given below in the “Megatron-axonn” section.
Step 4
Make your .bashrc in your home directory look like this:
```
[[ $- == *i* ]] || return
stty erase '^?'

alias cdme="cd /usr/workspace/$USER"

function mega_axonn() {
	for i in $(seq ${CONDA_SHLVL}); do
    		conda deactivate
	done
	export CONDAPATH=/usr/workspace/$USER/power_miniconda
	eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
	export PATH="/usr/workspace/$USER/power_miniconda/bin:$PATH"
	conda activate mega_axonn
}

if [ "$(hostname)" == "lassen710" ] || [ "$(hostname)" == "lassen709" ] || [ "$(hostname)" == "lassen708" ]; then
	export CONDAPATH=/usr/workspace/$USER/power_miniconda
	eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
	export PATH="/usr/workspace/$USER/power_miniconda/bin:$PATH"
	echo 'set up power conda'
fi

if [ "$(hostname)" == "pascal83" ] || [ "$(hostname)" == "tioga10" ] || [ "$(hostname)" == "tioga11" ]; then
  # Load conda (note: does not activate env):
  export CONDAPATH=/usr/workspace/$USER/x86_miniconda
  eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
  export PATH="/usr/workspace/$USER/x86_miniconda/bin:$PATH"
  echo 'set up x86 conda'
fi
```
Manual approach
The following steps will download the conda installation script to your workspace folder and start setting up conda.

cd /usr/workspace/$(whoami)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
mkdir power_miniconda
ml gcc/8.3.1 cuda/11.8.0 cmake/3.23.1 git/2.31.1
This is loading some basic dependencies that may be helpful
sh Miniconda3-latest-Linux-ppc64le.sh -b -p power_miniconda/

After going through the installation script and running `conda init` (it will prompt you to do this), add to conda’s config the OSU channel:

conda config --prepend channels https://ftp.osuosl.org/pub/open-ce/current/
You might also add the following channels, but i would keep osu’s at the top of your preference list: https://opence.mit.edu, https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/,https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access
When finished it should look like this (order matters)


Set up a new conda environment and `conda install` all the packages you want that are on the OSU channel (they are listed here: https://osuosl.org/services/powerdev/opence/). For other packages, use `pip install`.

Let me know if you run into issues! 
Debugging/Issues
If you get a CUDA version mismatch when using conda to install torch, you might need to load CUDA 11.8 on the system (even though we’re installing cudatoolkit 11.8 from conda)
run `module load cuda/11.8.0` then run your install command (e.g., `conda install pytorch=2.0.1`)

SSL problems you might find and potential fixes:
“SSLError: …”
It was discovered that this happens when conda installs openssl 3.X. This may be fixed by forcing conda to downgrade openssl to 1.1.1:
conda install --force-reinstall openssl=1.1.1
Other SSL errors
[recommended] conda config --set ssl_verify /etc/pki/tls/cert.pem
[if needed] conda config --set ssl_verify false

You might run into issues finding packages that work on Power9. Deepspeed, torch2, transformers, etc. all seem to work with conda install. However, triton does not seem to be working on Power9, limiting our ability to `torch.compile` models, for example.

Running jobs
Quick start
Just run python commands on the login nodes using:
The terminal
Jupyter
Login at https://lc.llnl.gov/jupyter/hub/
To get your custom conda environment visible in jupyter, activate your environment then run the following command in your home directory (the command might require you to `conda install ipykernel` first):
python -m ipykernel install --user
VScode
You can only ssh into tioga, not lassen, but the file systems are shared
Interactive jobs
A simple way to get 1 interactive node to yourself for 60 minutes:
lalloc 1
This uses bsub under the hood IIUC

Some examples of requesting interactive node(s) that use bsub’s many options:
bsub -q pbatch -nnodes 1 -Is -W 12:00 -G effml lexec
A bank name will be required when using the pbatch queue, which has a maximum job time limit of 12:00 hours. Our bank name is “effml”
bsub -q pdebug -nnodes 4 -Is -W 2:00 lexec
Differences relative to the previous command are highlighted
Note that we can request nodes through the pdebug queue for at most 2 hours
If you want nodes ASAP, using the pdebug queue (instead of pbatch) or reducing your requested time may help you.
Batch jobs
Don’t forget to activate your conda environment before submitting a batch job!

Submit a script that will be run on a node when the next node becomes available by using the following command:
bsub -q pbatch -nnodes 1 -W 12:00 -G effml -o logs/%J.out  python example.py
This assumes the directory “logs” exists, and the file logs/{job ID}.out will be where job print statements and error messages are written

Often, we will have several experiments we might want to run. We can loop over the various bsub commands corresponding to these individual experiments in a bash script `a_bunch_of_experiments.sh`, and submit all the experiments at once in the following way.
bsub < a_bunch_of_experiments.sh
Multinode jobs
You can request multiple nodes by setting “nnodes”>1 (e.g., see the multinode bsub command in the interactive section above).

However, to get python to see all the available nodes, you must prepend your python call with `jsrun` in the following way:
jsrun --smpiargs="-disable_gpu_hooks" -r $NUM_PROCESS_PER_NODE python example.py…
On 4-GPUs-per-node systems like Lassen, $NUM_PROCESS_PER_NODE will be 4 if you want 1 process per GPU
Documentation
There are tons of resources further describing usage of LSF and the LC system “Lassen”. https://hpc.llnl.gov/documentation/tutorials/using-lc-s-sierra-systems is fairly comprehensive. 

Slurm to LSF translator: https://portal.supercomputing.wales/index.php/index/slurm/lsf-to-slurm-ref/,  https://scicomp.ethz.ch/wiki/LSF_to_Slurm_quick_reference 

Megatron deepspeed (might be outdated, see my megatron-axonn section instead, or just use FSDP?)
https://lc.llnl.gov/confluence/display/LC/2021/12/14/Running+Megatron-DeepSpeed+with+Distributed+PyTorch+on+Lassen 

Megatron-axonn 
The following script is available on github, which specifies the branch to use after cloning axonn and megatron-axonn https://github.com/jwkirchenbauer/llm-pretraining/blob/main/power9_megatron_axonn_setup.sh 

It will take an hour to set up the environment by running `source power_megatron_axonn_setup.sh`, where the shell script is:
```
conda create -y -n apex
conda activate apex
ml cuda/11.8.0 gcc/11.2.1
conda install -y  pytorch=2.0.1 cudatoolkit=11.8.0
conda install -y  gxx_linux-ppc64le
conda install -y  ninja
conda install -y  psutil pybind11
git clone https://github.com/NVIDIA/apex
pushd apex
	# checkout this branch based on https://github.com/NVIDIA/apex/issues/1735
	git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
	# this is going to take almost an hour to build
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
git clone git@github.com:axonn-ai/axonn.git
pushd axonn
	pip install -e .
popd
conda install -y six regex
```
Use the submission script here (after modifying directory arguments like DATA_DIR): https://github.com/jwkirchenbauer/llm-pretraining/blob/main/LSF_axonn.sh

When submitting, you will likely get an error that can be fixed: I added False for the missing positional parameter in megatron/model/fused_layer_norm.py (siddharth mentioned that john did this originally).

Tioga (x86, MI250X)
Each tioga node has 8 GPUs.

WARNING: there are only ~20 tioga nodes total. Try not to use too many of them for long periods of time. On weekends or evenings when no one else is on, using ~16 should be fine.
Helpful commands to see who is running things and what resources are available:
flux jobs -u all
flux resource list 

WARNING 2: tioga is a test system with changing software and potential bugs. It was designed to help us figure out issues before its larger version El Capitan (which will have thousands of nodes) goes live in 2024. 
Logging in
ssh username@tioga.llnl.gov (does not require connecting to the VPN)
ssh username@tioga (requires connecting to the VPN)
Anaconda setup
Tioga uses x86, so there are no special channels needed for anaconda/pip. Just download things as you normally would. However, it is using AMD GPUs, and some packages will be modified for rocm’s use instead of cuda’s. 
For example, for pytorch, i have been pip installing the following wheel torch-2.0.1+rocm5.4.2-cp311-cp311-linux_x86_64.whl 
I suggest finding a wheel matching this pattern if you install torch 2.1 instead of torch 2.0.1

Assuming you set up a Power9 anaconda environment on Lassen, you will need a separate x86 anaconda installation on Tioga.
To keep them separate, you must not run `conda init` when you are prompted during the x86 conda installation. 
Instead of running conda init, modify your bashrc to set up the x86 conda as needed after installing conda (see below for details)

For example, do something like the following:
Install miniconda in /usr/workspace/$USER/x86_miniconda
cd /usr/workspace/$(whoami)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
mkdir x86_miniconda
sh Miniconda3-latest-Linux-x86_64.sh -f -p x86_miniconda
DON’T RUN CONDA INIT WHEN IT ASKS DURING INSTALLATION
In the terminal, paste the following into your ~/.bashrc file:

 if [ "$(hostname)" == "tioga10" ] || [ "$(hostname)" == "tioga11" ]; then
  # Load conda (note: does not activate env):
  export CONDAPATH=/usr/workspace/$USER/x86_miniconda
  eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
fi
if [ "$(hostname)" == "pascal83" ] || [ "$(hostname)" == "tioga10" ] || [ "$(hostname)" == "tioga11" ] || [[ "$(hostname)" == *"tuolumne"* ]]; then
  # Load conda (note: does not activate env):
  export CONDAPATH=/usr/workspace/$USER/power9_miniconda
  eval "$(${CONDAPATH}/bin/conda shell.bash hook)"
fi
If you have a power9 conda install, then your bashrc will likely also have other conda setup code, which you can wrap inside a similar if condition that checks if $CLUSTER == “lassen” 
Running jobs
Note that tioga does not require a compute bank (“effml” for us). You can just run jobs for free.
Quick start
Like lassen, tioga has GPUs on the login nodes that you can use to run small tests. Login via vscode or just use the terminal and submit python commands to get things running right away.
Interactive jobs
To get your own interactive nodes, use `flux alloc`. For example, to request an interactive node with 16 processes/ranks spread across 2 nodes, use 
`flux alloc -n16 -N2`.

Note that we have 16 ranks because each node has 8 GPUs.

Helpful commands:
flux resource list
flux jobs -u all
flux cancel
watch -n 0.01 /opt/rocm-5.4.2/bin/rocm-smi  --showmeminfo vram /opt/rocm-5.4.2/bin/rocm-smi --showmeminfo vram
Batch
We can use the flux batch command as shown in https://github.com/Ping-C/llama_cml/blob/amd-adv-training/job_chainer.sh. i.e.

```
#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <unique job name>"
  exit 1
fi
job_name=$1

# max number of jobs to chain.
export MAX=5
# script with specific environment settings for the job
export JOB_SCRIPT=amd_submit.sh
# time limit for the job, HH:MM:SS format
export TIME=600
export NODES=8
export PROCS=64
#export NODES=4
#export PROCS=32

# Create a normal job without a dependency
export currcount=0
name="${job_name}_${currcount}"
job_id=$(flux batch -t $TIME -N $NODES -n $PROCS --output=logs/$name $JOB_SCRIPT)

# create MAX-1 more jobs that depend on the previous job
export currcount=$(expr $currcount + 1)
while [ $currcount -lt $MAX ]
do
    name="${job_name}_${currcount}"
    job_id=$(flux batch -t $TIME -N $NODES -n $PROCS --output=logs/$name --dependency=afterany:$job_id $JOB_SCRIPT)
    export currcount=$(expr $currcount + 1)
done
```
Multinode jobs
To get python to see all available nodes/ranks, prepend your python call with `flux run` as shown in https://github.com/Ping-C/llama_cml/blob/amd-adv-training/amd_submit.sh. 
Documentation
Tioga uses the job manager “flux”, which can work with a lot of slurm commands (I think your slurm scripts might just work on tioga, assuming you add the AMD boilerplate shown in https://github.com/Ping-C/llama_cml/blob/amd-adv-training/amd_submit.sh). 

Here is some flux documentation: https://flux-framework.readthedocs.io/en/latest/quickstart.html#manual-installation. 

Flash attention
From siddharth:
Now we will install from https://github.com/ROCmSoftwarePlatform/flash-attention
clone it
open setup.py and replace c++20 by c++17
https://github.com/ROCmSoftwarePlatform/flash-attention/blob/flash_attention_for_rocm/setup.py#L257
https://github.com/ROCmSoftwarePlatform/flash-attention/blob/flash_attention_for_rocm/setup.py#L260
PYTORCH_ROCM_ARCH='gfx90a' GPU_ARCHS='gfx90a' pip install .


NERSC
Quick Start
go to jupyter.nersc.gov/

select “login node”

make your own conda environment, or to quickly try to run something, activate one i made:
module load conda/Miniconda3-py311_23.11.0-2
conda activate /global/common/software/m4536/tool_misuse

use the jobscript generator (https://iris.nersc.gov/utils/jobscript), or see the following job-launch example for an illustration of key variables

Create the files `sbatch.sh`, `launcher.sh`, and `test_script.py`, activate your conda environment, then run `sh sbatch.sh`. You can view the output in the file “test” that gets created in your current directory.
Note that this example illustrates multi-node usage and may be more complex than what you need

sbatch.sh = 
```
#!/bin/bash

# bank to use for the job allocation
export ACCOUNT=m4536
# QOS
export QOS=debug #regular
# time limit for the job, HH:MM:SS format
export TIME=00:05:00
# GPU Constraint. Can be one of [gpu, gpu_hbm80g]
export GPU_CONSTRAINT=gpu
# N Nodes
export NNODES=2
# Name
export name=test

sbatch  --qos $QOS --nodes $NNODES --time $TIME --constraint $GPU_CONSTRAINT --gpus-per-node 4 --account $ACCOUNT -o $name --wrap="sh launcher.sh"
``

launcher.sh =
```
# Generate a random port number to avoid conflicts
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
export MASTER_ADDR=$(hostname)
export NUM_PROCESS_PER_NODE=4
export WORLD_SIZE=$(($NUM_PROCESS_PER_NODE*$SLURM_NNODES))
echo "NUM_PROCESS_PER_NODE="${NUM_PROCESS_PER_NODE}
echo "WORLD_SIZE="${WORLD_SIZE}
echo "NODELIST="${SLURM_NODELIST}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT

nvidia-smi
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE
echo "SLURM_GPUS_PER_NODE="$SLURM_GPUS_PER_NODE


srun --ntasks-per-node $NUM_PROCESS_PER_NODE python test_script.py
```

test_script.py = 
```
import torch
import torch.distributed as dist
import os

def main():
	world_size = int(os.environ['WORLD_SIZE'])
	rank = int(os.environ['SLURM_PROCID'])
	dist.init_process_group(backend='nccl',
                        	init_method='env://',
                       	world_size=world_size,
                       	rank=rank)
	tensor = torch.tensor([rank])
	print(f"Rank {rank} has tensor: {tensor}")
	dist.destroy_process_group()

if __name__ == "__main__":
	main()
```
Details
each node has 4xA100 (80 GB)

try jupyter.nersc.gov. you should be able to access A100 nodes there.
you can go to a login node and use slurm commands to launch jobs, or (if you just need 1 node with 4 GPUs) use the exclusive GPU node option instead
the login node use is free, but launching jobs with slurm commands or using that exclusive GPU node will cost node-hours, which are limited

you can use the terminal to log in (https://docs.nersc.gov/connect/) if you prefer that to the web GUI

there’s currently a limit of about 1500 node hours per person
we can get you more if needed
if you’re having trouble getting nodes, then please let me know
we can get you priority access or show you some things that might help

iris.nersc.gov might be useful to see your node hour usage

docs:
https://docs.nersc.gov/jobs/

you can request many nodes per job (e.g. 250), but the more nodes you request, the longer it’ll take your job to be allocated
also, if you request 250 nodes for 6 hours total, that would burn 1500 node hours (your entire allocation) with one job


LLnL
Jul 18, 2024
Brian and Christian

LLNL Environment Creation and Job Running

Brian, AI Safety work, robustness of vision and language models - AMD GPUs
LCAP comces online soon, MI300
Tioga system has a few nodes with MI300s
Separate LC username
8 character pin 6 digit token 
Add or remove users
https://hpc.llnl.gov/accounts/idm/getting-started
https://hpc.llnl.gov/accounts/idm/users/how-be-added-resource
https://lc-idm.llnl.gov/request

Waiting approval 




tomaschke1




Welcome to the OSLIC Cluster,
Oslic

Here’s a concise step-by-step guide for each time you log in to run your code on Tioga:
1. SSH into Tioga via VS Code
	1.	Open VS Code.
	2.	Use the Remote-SSH extension to connect to Tioga:
	•	Press F1, type “Remote-SSH: Connect to Host”, and select tioga.
2. Check Available Resources
	1.	Open a terminal in VS Code (Ctrl+` or View > Terminal).
	2.	Check available resources and running jobs:
flux resource list
flux jobs -u all
3. Navigate to Your Project Directory
	1.	Change to your project directory:
cd /usr/workspace/smith585/codebases/collaborative-stegosystem
4. Activate Your Conda Environment
	1.	Activate the Conda environment for your project:
conda activate collaborative-stegosystem
5. Allocate Interactive Node (Optional for Testing)
	1.	Allocate an interactive node if needed for testing:
flux alloc -n1 -N1
	2.	Run your script interactively:
python your_script.py
6. Prepare and Submit a Batch Job
	1.	Create a job script (job_script.sh):
nano job_script.sh
Add the following content to job_script.sh:
#!/bin/bash
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 2:00
#BSUB -J collaborative_stegosystem_job
#BSUB -o logs/%J.out

# Load environment
source ~/.bashrc
conda activate collaborative-stegosystem

# Navigate to project directory
cd /usr/workspace/smith585/codebases/collaborative-stegosystem

# Run your script
python your_script.py


chmod +x job_script.sh

flux submit -N 1 -n 8 bash job_script.sh


	2.	Submit the batch job:
flux batch job_script.sh
	1.	Monitor the status of your job:
flux jobs -u all
	2.	View output logs (replace <job_id> with your actual job ID):
cat logs/<job_id>.out
	1.	Cancel a running job (replace <job_id> with your actual job ID):
flux cancel <job_id>

Jupyter notebook
https://hpc.llnl.gov/search/node?keys=jupyter

https://lc.llnl.gov/orbit/gravitate/tioga10/hub/spawn

Clean disk space
~/.cache - rm hugging face 
du -s * | sort -nr | head -n10

Check usage
/opt/rocm-5.4.2/bin/rocm-smi

Accelerate config
Accelerate launch file.py

Talk to support about ip addresses linking on multiplenodes
Rocm torch on cuda env



Run Dust
https://github.com/schollz/croc


