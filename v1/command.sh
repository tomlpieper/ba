srun -K \
  --job-name="TrialT5" \
  --gpus=1 \
  --mem=32G \
  --container-mounts=/netscratch/tpieper:/netscratch/tpieper,"`pwd`":"`pwd`" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
  --container-save=/netscratch/tpieper/containers/custom_pytorch_env.sqsh \
  --container-workdir="`pwd`" \
  /netscratch/tpieper/install.sh