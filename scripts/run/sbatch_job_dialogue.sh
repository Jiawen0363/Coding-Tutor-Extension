#!/usr/bin/env bash
# 仅提交用：dialogue（1 或 2 卡）。卡数由 submit 脚本 --gres=gpu:N 覆盖；实现逻辑在 sbatch_run_test.sh。
#SBATCH --job-name=test-dialogue
#SBATCH --output=/data/%u/slurm-logs/%x-%j.out
#SBATCH --error=/data/%u/slurm-logs/%x-%j.err
#SBATCH --gres=gpu:2
# 1 GPU 作业在当前集群普通队列上限为 12 CPU。
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=8:00:00

export TEST_PIPELINE_PHASE=dialogue
[ -n "${PROJECT_ROOT:-}" ] || { echo "错误: PROJECT_ROOT 未设置"; exit 1; }
# OUTPUT_DIALOGUE_BASE / OUTPUT_STUDENT_POSTTEST_BASE 由 submit_test.sh 可选 export；不设置时 run_*.sh 用默认 output/dialogue 与 output/student_posttest
GPU_BIND="${SLURM_GPU_BIND:-map_gpu:0}"
exec srun --ntasks=1 --gpu-bind="$GPU_BIND" bash "$PROJECT_ROOT/scripts/run/sbatch_run_test.sh"
