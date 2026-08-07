# Setup and Run data preprocessing on HPC Eagle Cluster

Runs the data preprocessing on the HPC Eagle cluster.

# Setup (one-time per user)

Created originally by Adam Mazur, modified by Me

Eagle is a SLURM based system, which means you have to allocate the compute before you use it. Here are the PCSS docs on how to use SLURM: https://help.pcss.plcloud.pl/portal/hpc/4%20Job%20Management%20and%20Scheduling/#scheduler-overview

## Using VSCode on Eagle

When you SSH into Eagle, you will land on a login node, which is very very slow, so don’t try to connect VSC directly to Eagle with SSH. Please modify your SSH config (`C:\Users\<USER>\.ssh\config`) by pasting this (***WARNING: this only works on Windows, ask GPT to make it work on Linux***):

```
Host eagle.man.poznan.pl
  HostName eagle.man.poznan.pl
  IdentityFile C:\Users\<USER>\.ssh\id_ed25519
  User YOUR_USERNAME_HERE
  Compression yes
  TCPKeepAlive yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
  ControlMaster auto
  ControlPersist 600
  Ciphers chacha20-poly1305@openssh.com,aes128-ctr

Host vscode-eagle
    User YOUR_USERNAME_HERE
    ForwardAgent yes
    StrictHostKeyChecking no
    UserKnownHostsFile NUL
    ServerAliveInterval 30
    ProxyCommand ssh eagle.man.poznan.pl "/usr/bin/salloc -A pl1157-01 -J vscode -p interactive -N1 -n1 -c4 --mem=8G -t 8:00:00 bash -c 'exec nc $(scontrol show hostnames $SLURM_NODELIST | head -1) 22'"
```

**Please switch the username from `YOUR_USERNAME_HERE` to your username, and change the paths to make them work on your machine.**

Then to connect VSC to Eagle simply connect to vscode-eagle, instead of directly connecting to Eagle.

## Setting up the packages

When you first login on Eagle, your home directory will have a maximum space of 1GB!!!

This means you cannot install anything on your home dir, you need to move all the config files to our project’s storage. 

- Create your own directory on the shared project storage:
    
    ```
    cd ~
    mkdir <GRANT>/project_data/$USER
    ```
    
- Create directories for all config’s and caches that libraries like uv use:
    
    ```
    cd ~
    mkdir <GRANT>/project_data/$USER/.cache
    mkdir <GRANT>/project_data/$USER/.config
    mkdir <GRANT>/project_data/$USER/.ipython
    mkdir <GRANT>/project_data/$USER/.local
    ```
    
- Then create symlinks to those directories in your home directory:
    
    ```
    cd ~
    ln -s <GRANT>/project_data/$USER/.cache .cache
    ln -s <GRANT>/project_data/$USER/.config .config
    ln -s <GRANT>/project_data/$USER/.ipython .ipython
    ln -s <GRANT>/project_data/$USER/.local .local
    ```
    
- After you do that you can proceed by installing uv, claude, etc.

## Setting up the repo

You need to setup the repo in the `<GRANT>/project_data/$USER` directory, please don’t use any other directory.

- Clone the repo (you might need to generate an SSH key and add it to your GitHub to do this):
    
    ```
    cd <GRANT>/project_data/$USER
    git clone 
    ```
    
- Setup uv:
    
    ```
    UV_LINK_MODE=hardlink uv sync
    ```
    
    Remember about using `UV_LINK_MODE`! That’s important.
    

## HuggingFace Authentication (one-time per user)

The LLAMA3.1 model is gated — you need a HuggingFace account with access granted.

1. Request access at: https://huggingface.co/meta-llama/LLAMA3.1-8B
2. Generate a read token at: https://huggingface.co/settings/tokens
3. Create a `.env` file and add the HF_TOKEN (see .env.example)

## Running the Setup

If you want to run that evaluation for first time, you have to install dependencies.
Navigate to the evaluation directory and run setup.sh script, which installs those dependencies and creates virtual environment for you.

```bash
bash data_preprocessing/pcss_use/setup.sh
```

# Evaluation

## Running the Evaluation

Navigate to the evaluation directory and submit the job:

```bash
sbatch data_preprocessing/pcss_use/run_eval.sh
```


---

## Monitoring

```bash
# Check job status
squeue -u $USER

# Follow logs in real time (replace JOBID with your job number)
tail -f logs/JOBID.out

# Check errors
cat logs/JOBID.err
```

---

## Project Structure

```
$PROJECT
├── data_preprocessing
│   ├── pcss_use
│   │   ├── README.md
│   │   ├── setup.sh
│   │   ├── run_eval.sh
│   ├── requirements.txt
│   ├── pipeline.py
│   ├── .env.example
│   └── ...
```

## Troubleshooting

**Job pending (`PD`)** — normal, waiting for a free GPU. Check with `squeue -u $USER`.

**`ModuleNotFoundError: No module named 'lm_eval'`** — venv is broken, delete it and reinstall:
```bash
rm -rf $PROJECT_PATH/venv
cd $PROJECT_PATH/evaluation
bash setup.sh
sbatch run_benchmark.sh
```

**`Disk quota exceeded`** — check usage:
```bash
du -sh $PROJECT_PATH/*
```

**Slow installation of dependencies** — I don't know why this happened for me, but if you have a slow installation of dependencies, you may try to restart the script; it suprisingly helped me.