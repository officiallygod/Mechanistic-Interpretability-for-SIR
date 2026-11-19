#!/bin/bash
#SBATCH -p accelerated
#SBATCH -A hk-project-p0024638
#SBATCH -J TrainModels
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/hk-project-p0024638/wx3982/logs/%x_%A_%a.out
#SBATCH --error=/home/hk-project-p0024638/wx3982/logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=0 # Use all available memory (only recommended for image baseline models)

# -------------------------------
# Environment setup
source ~/.bashrc
conda activate robocasa_mdt

echo "TMPDIR is set to $TMPDIR"

echo "Running on $(hostname)"
echo "Starting time: $(date)"

BASE_DATA_PATH="/hkfs/work/workspace/scratch/wx3982-datasets/robocasa"

echo "Copy dataset to TMPDIR: $TMPDIR"
mkdir -p $TMPDIR/datasets
for task_name in OpenSingleDoor OpenDoubleDoor CloseSingleDoor CloseDoubleDoor \
                     OpenDrawer CloseDrawer TurnOnStove TurnOffStove TurnOnSinkFaucet \
                     TurnOffSinkFaucet TurnSinkSpout CoffeePressButton TurnOnMicrowave \
                     TurnOffMicrowave CoffeeServeMug CoffeeSetupMug PnPCabToCounter \
                     PnPCounterToCab PnPCounterToSink PnPCounterToStove PnPMicrowaveToCounter \
                     PnPSinkToCounter PnPStoveToCounter PnPCounterToMicrowave; do
    
    SRC_DIR="$BASE_DATA_PATH/${task_name}"
    DEST_DIR="$TMPDIR/datasets/${task_name}"

    mkdir -p $DEST_DIR
    rsync -av $SRC_DIR/demo_gentex_im128_randcams.hdf5 $DEST_DIR/
    rsync -av $SRC_DIR/bb_coordinates_left_image.pth $DEST_DIR/
    rsync -av $SRC_DIR/bb_coordinates_right_image.pth $DEST_DIR/

    # Comment this out if image tensors are not needed
    echo "Copying image tensors for $task_name..."
    for ((img_idx=0; img_idx<=54; img_idx++)); do
        
        SRC_FILE="$SRC_DIR/img_tensor_demo_${img_idx}.pth"
        
        # Check if the specific file exists before trying to copy it
        # This prevents 'file not found' errors from rsync
        if [ -f "$SRC_FILE" ]; then
            rsync -av $SRC_FILE $DEST_DIR/
        fi
    done
done

echo "Dataset copied."

SCRIPT_PATH=/home/hk-project-sustainebot/wx3982/repos/robocasa_mdt/train.py
LOG_PATH=/home/hk-project-p0024638/wx3982/runs/

# Start two training processes in parallel, each pinned to a different GPU and logging to its own file.
echo "Starting training processes..."
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_PATH}" log_dir="${LOG_PATH}" manager.data_path="${TMPDIR}/datasets/" manager.graph_dataset_path="${TMPDIR}/datasets/" seed=101010

echo "Process $pid1 exited with code $rc1"

if [ "$rc1" -ne 0 ]; then
    echo "One or more training processes failed (rc1=$rc1)" >&2
    exit 1
fi


echo "Ending time: $(date)"
