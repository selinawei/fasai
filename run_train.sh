#!/bin/bash




#$ -pe smp 4  # Specify parallel environment and legal core size
#$ -q gpu@@crc_gpu      # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N LLM_chat      # Specify job name


conda activate wlx
python3 /afs/crc.nd.edu/user/l/lwei5/Private/FEA-SAI/train.py
