#!/bin/bash
#PBS -l select=8:ncpus=80:mem=100gb
#PBS -q development
#PBS -l walltime=06:00:00
#PBS -N restartable_sim
#PBS -o job_output.log
#PBS -e job_error.log
#PBS -V

cd $PBS_O_WORKDIR

# Debug: print working directory and contents
echo "===============================================" >> full_simulation.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running in: $(pwd)" >> full_simulation.log
ls -l >> full_simulation.log
echo "===============================================" >> full_simulation.log

# Load environment
everything

# --- Set default step ---
starting_step=0

# --- Find latest timestep (from rank 0 files only) ---
for file in result/state_vector_*_000.*.npy; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        step_str="${filename##*.}"
        step=$((10#$step_str))
        if [ "$step" -gt "$starting_step" ]; then
            starting_step=$step
        fi
    fi
done

# --- Stop if max step reached ---
MAX_STEP=800
if [ "$starting_step" -ge "$MAX_STEP" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Maximum step $MAX_STEP reached. Simulation finished." >> full_simulation.log
    exit 0
fi

# --- Update config ---
sed -i "s/^starting_step *= *.*/starting_step = $starting_step/" config/hydrostaticAtmosphere.ini

# --- Append timestamp and step info to log ---
echo "===============================================" >> full_simulation.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting simulation from step $starting_step" >> full_simulation.log
echo "===============================================" >> full_simulation.log

# --- Run simulation and append both stdout + stderr to one file ---
pwd
./main.py config/hydrostaticAtmosphere.ini >> full_simulation.log 2>&1

# --- Resubmit the job ---
# qsub $0

