#!/bin/bash


#$ -M lwei5@nd.edu
#$ -m abe
#$ -pe smp 2  # Specify parallel environment and legal core size
#$ -q gpu@qa-a10-026      # Run on the GPU cluster
#$ -l gpu_card=2     # Run on 1 GPU card
#$ -N LLM_chat      # Specify job name


conda activate wlx
python3 /afs/crc.nd.edu/user/l/lwei5/Private/FEA-SAI/model/__init__.py
