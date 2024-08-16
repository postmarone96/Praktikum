# Model
MODEL=$(jq -r '.Metrics.model' params.json)

# Project Directory
P_DIR=$(jq -r '.project_dir' params.json)
export P_DIR


test -d metrics_${MODEL} || mkdir metrics_${MODEL}
cp -f cal_metrics.slurm metrics/
