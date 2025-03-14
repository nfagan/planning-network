LOCAL_DIR="../checkpoints"
REMOTE_DIR="naf264@greene.hpc.nyu.edu:/scratch/naf264/planning-network/checkpoints"

echo "Uploading new files to remote..."
rsync -avz --ignore-existing "${LOCAL_DIR}/" "${REMOTE_DIR}"

echo "Downloading new files from remote..."
rsync -avz --ignore-existing "${REMOTE_DIR}/" "${LOCAL_DIR}"