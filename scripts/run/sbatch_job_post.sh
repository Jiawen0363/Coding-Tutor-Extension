#!/usr/bin/env bash
# 仅提交用：单卡 post（code_gen + coding_test）。实现逻辑在 sbatch_run_test.sh。
#SBATCH --job-name=test-post
#SBATCH --output=/data/%u/slurm-logs/%x-%j.out
#SBATCH --error=/data/%u/slurm-logs/%x-%j.err
#SBATCH --gres=gpu:1
# 单卡作业在当前集群普通队列通常上限为 12 CPU
#SBATCH --cpus-per-task=12
# 单卡作业在当前集群普通队列通常上限为 60G 内存
#SBATCH --mem=60G
#SBATCH --time=48:00:00

export TEST_PIPELINE_PHASE=post
[ -n "${PROJECT_ROOT:-}" ] || { echo "错误: PROJECT_ROOT 未设置"; exit 1; }
# OUTPUT_DIALOGUE_BASE / OUTPUT_STUDENT_POSTTEST_BASE 由 submit_test.sh 可选 export；不设置时 run_*.sh 用默认 output/dialogue 与 output/student_posttest
exec bash "$PROJECT_ROOT/scripts/run/sbatch_run_test.sh"
