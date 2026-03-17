#!/bin/bash
#SBATCH --job-name=ray_tune_multinode_jeanzay
#SBATCH -C h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=00:06:00
#SBATCH --exclusive
#SBATCH --output=/lustre/fsn1/projects/rech/name-project/jean-zay-id/path/to/logs/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/name-project/jean-zay-id/path/to/logs/%x_%j.err
#SBATCH -A ame-project@h100
#SBATCH --signal=B:SIGUSR1@120  # Envoie le signal SIGUSR1 60 secondes avant la fin

set -x

# Handler trigered 120s before the ending by Slurm
function propagate_signal() {
    echo "Signal SIGUSR1 traped at T-120s. Sending SIGINT to Ray..." 
    kill -SIGINT $PYTHON_PID || true
}
trap propagate_signal SIGUSR1

#############################################
# Software environnement
#############################################

# Loading the virtual environnement
module purge
cd $SCRATCH/hpo_raytune
source venv_hpo_raytune/bin/activate
cd multi_node

# Displaying the job information
scontrol show job $SLURM_JOB_ID

echo "========================================" 
echo "Starting Ray Tune job" 
echo "Job ID: $SLURM_JOB_ID" 
echo "Timestamp: $(date)" 
echo "Time limit: $SLURM_TIMELIMIT" 
echo "========================================" 


############################################
# retrieving the nodes
############################################

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${nodes[0]}
HEAD_IP=$(getent hosts $HEAD_NODE | awk '{print $1}')

echo "Nodes allocated: ${nodes[@]}"
echo "Head node: $HEAD_NODE ($HEAD_IP)"


############################################
# Launching the Ray head on the head node

############################################
echo "Starting Ray head node..."
srun --nodes=1 --ntasks=1 -w $HEAD_NODE \
    ray start --head \
    --node-ip-address=$HEAD_IP \
    --port=6379 \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --disable-usage-stats \
    --block &

HEAD_PID=$!
sleep 10


############################################
# Launching the Ray workers on the node children
############################################

WORKER_PIDS=()
for node in "${nodes[@]:1}"; do
    WORKER_IP=$(getent hosts $node | awk '{ print $1 }')
    echo "Starting Ray worker on $node ($WORKER_IP)"

    srun --nodes=1 --ntasks=1 -w $node \
        ray start \
        --address=$HEAD_IP:6379 \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --disable-usage-stats \
        --block &
    
    WORKER_PIDS+=($!)
    sleep 5
done

echo "All Ray nodes started"
echo "Head PID: $HEAD_PID"
echo "Worker PIDs: ${WORKER_PIDS[@]}"


############################################
# Launching Ray tune python script on the head node 
#   |_ The node is responsible for spreading the trials across the workers
############################################

echo "========================================" 
echo "Starting Ray Tune optimization" 
echo "Timestamp: $(date)" 
echo "========================================" 

sleep 10
python -u ray_tune.py --config config.yaml &

PYTHON_PID=$!

# Deactivating set -e to avoid slrum script to crash after before the wait
set +e
wait $PYTHON_PID
RC=$?
set -e

# If the wait returned a code > 128 (e.g., 138), it was interrupted by our "propagate_signal" trap.
# Python is therefore saving using SIGINT. You need to restart the wait to wait for it to finish.
if [ $RC -gt 128 ]; then
    echo "Wait interrompu par Slurm. Attente de la fin de la sauvegarde de Ray Tune..." 
    set +e
    wait $PYTHON_PID
    RC=$?
    set -e
fi

echo "Ray Tune finished with exit code: $RC"


echo "========================================"
echo "Ray Tune finished with exit code: $RC"
echo "Timestamp: $(date)"
echo "========================================"


############################################
# Ending logic
############################################
if [ $RC -eq 12 ]; then
    echo "Ray Tune incomplete (exit code 12) → should automatically requeue Slurm job to complete remaining trials!" 
    #scontrol requeue $SLURM_JOB_ID
    exit 0
elif [ $RC -eq 0 ]; then
    echo "Ray Tune completed successfully → No need to automatically requeue the job!"
    exit 0
else
    echo "Ray Tune failed with exit code $RC → Check logs for errors" 
    echo "NOT requeueing due to error" | tee -a $RAY_LOG_FILE
    exit $RC
fi



echo "Ray Tune completed !"
ray stop