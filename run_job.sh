#!/bin/bash
#PBS -N dcmip31
#PBS -l select=1:ncpus=24:mpiprocs=24:mem=100G
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o dcmip31_pbs.log

set -eo pipefail

# ============================================================
# User settings
# ============================================================

WORKDIR="/home/shp000/site8/U2_data/raid/ppp6/deviationMethod"
CONFIG_FILE="config/dcmip31.ini"
NUM_MPI_RANKS=6
LOG_FILE="${WORKDIR}/live.log"

CONDA_ROOT="/home/shp000/site8/conda/miniforge3"

# ============================================================
# Reproduce the commands inside wxenv_old
# ============================================================

export QT_QPA_PLATFORM=offscreen

source \
"/fs/ssm/main/opt/intelcomp/master/inteloneapi_2022.1.2_multi/oneapi/compiler/latest/env/vars.sh"

source \
"/fs/ssm/main/opt/intelcomp/master/inteloneapi_2022.1.2_multi/oneapi/mpi/latest/env/vars.sh"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate gef_310

# ============================================================
# Run
# ============================================================

cd "${WORKDIR}"

{
    echo "============================================================"
    echo "Job ID:       ${PBS_JOBID:-unknown}"
    echo "Host:         $(hostname)"
    echo "Working dir:  $(pwd)"
    echo "Python:       $(which python)"
    echo "MPI launcher: $(which mpirun)"
    echo "MPI ranks:    ${NUM_MPI_RANKS}"
    echo "Config:       ${CONFIG_FILE}"
    echo "Start time:   $(date)"
    echo "============================================================"
} > "${LOG_FILE}"

mpirun -n "${NUM_MPI_RANKS}" \
    "$(which python)" -u \
    main_gef.py \
    "${CONFIG_FILE}" \
    >> "${LOG_FILE}" 2>&1

run_status=$?

{
    echo "============================================================"
    echo "Finished:    $(date)"
    echo "Exit status: ${run_status}"
    echo "============================================================"
} >> "${LOG_FILE}"

exit "${run_status}"