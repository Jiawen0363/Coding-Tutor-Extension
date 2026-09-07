#!/usr/bin/env bash

# 提交作业：
#
# RUN_DIALOGUE_JOB=1:
#   [EXTRA_DEP_JOB] -> dialogue -> post
#
# RUN_DIALOGUE_JOB=2:
#   dialogue only
#
# RUN_DIALOGUE_JOB=0:
#   [EXTRA_DEP_JOB] -> post only
#
# dialogue GPU 数量由 TUTOR_VLLM_BACKEND 决定：
#   remote/API        -> 1 GPU
#   qwen/llama local  -> 2 GPUs

set -euo pipefail


# ============================================================
# 基础配置
# ============================================================

ROOT="/data/wangjian/Coding-Tutor-Extension"

export PROJECT_ROOT="$ROOT"

mkdir -p "/data/${USER}/slurm-logs"


# ============================================================
# 输出路径
# ============================================================

# 如果保留下面两行：
# dialogue / post 输出到 validate_reward。
#
# 如果注释掉：
# 使用项目默认路径：
#   output/dialogue
#   output/student_posttest

# export OUTPUT_DIALOGUE_BASE="$ROOT/validate_reward/output/dialogue"
# export OUTPUT_STUDENT_POSTTEST_BASE="$ROOT/validate_reward/output/student_posttest"


# ============================================================
# 实验配置
# ============================================================

export ITERATION="first_iter"

export TUTOR_VLLM_BACKEND="qwen"

export TUTOR_MODEL="Qwen3-8B-opd-1787576349"

export BASE_MODEL_PATH="/data_old/models/Qwen3-8B"

export LORA_ADAPTER="/data/wangjian/Coding-Tutor-Extension/checkpoints/OPD/Qwen3-8B-1787576349/checkpoint-135"


# ============================================================
# Dialogue 阶段 vLLM 配置
# ============================================================

# run_base 会访问本节点启动的 tutor / student HTTP vLLM。
#
# post 阶段 run_code_gen_new.sh 使用 LM_inference.py
# 进程内加载模型，不使用这些 HTTP 端口。

export TUTOR_VLLM_PORT="8001"
export STUDENT_VLLM_PORT="8002"

export STUDENT_MODEL_PATH="/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ"


# ============================================================
# Dialogue GPU 配置
# ============================================================

GPU_CONFIG="$ROOT/scripts/run/configure_dialogue_gpus.sh"

if [[ ! -f "$GPU_CONFIG" ]]; then
    echo "ERROR: GPU config script not found:"
    echo "  $GPU_CONFIG"
    exit 1
fi

# shellcheck source=scripts/run/configure_dialogue_gpus.sh
source "$GPU_CONFIG"

# 防止 configure_dialogue_gpus.sh 没有正确设置变量。
: "${DIALOGUE_SBATCH_GRES:?ERROR: DIALOGUE_SBATCH_GRES was not set by configure_dialogue_gpus.sh}"
: "${SLURM_GPU_BIND:?ERROR: SLURM_GPU_BIND was not set by configure_dialogue_gpus.sh}"

echo "TUTOR_VLLM_BACKEND=$TUTOR_VLLM_BACKEND"
echo "DIALOGUE_SBATCH_GRES=$DIALOGUE_SBATCH_GRES"
echo "SLURM_GPU_BIND=$SLURM_GPU_BIND"


# ============================================================
# 可选 chat template
# ============================================================

# 如果 Qwen3 需要 non-thinking template，可以打开：
#
export TUTOR_CHAT_TEMPLATE="$ROOT/scripts/run/qwen3_nonthinking.jinja"


# ============================================================
# Job 模式
# ============================================================

# 1 = dialogue -> post
# 2 = dialogue only
# 0 = post only
#
# 如果外部已经设置 RUN_DIALOGUE_JOB，则尊重外部值。
# 否则默认 1。

export RUN_DIALOGUE_JOB="${RUN_DIALOGUE_JOB:-1}"


# ============================================================
# 可选额外 dependency
# ============================================================

# 不要在这里硬编码 1239。
#
# 默认没有额外 dependency：
#
#   bash scripts/run/submit_test0.sh
#
# 如果需要先等某个已有 Slurm job：
#
#   EXTRA_DEP_JOB=1239 bash scripts/run/submit_test0.sh
#
# 或：
#
#   export EXTRA_DEP_JOB=1239
#   bash scripts/run/submit_test0.sh

export EXTRA_DEP_JOB="${EXTRA_DEP_JOB:-}"


# ============================================================
# 辅助函数
# ============================================================

check_extra_dependency() {
    if [[ -z "${EXTRA_DEP_JOB:-}" ]]; then
        return 0
    fi

    # 这里只验证格式。
    # 不强制使用 scontrol 检查，因为某些集群对已完成 job
    # 的 scontrol 查询行为不同，但 Slurm dependency 仍可能合法。
    if [[ ! "$EXTRA_DEP_JOB" =~ ^[0-9]+$ ]]; then
        echo "ERROR: EXTRA_DEP_JOB must be a numeric Slurm job ID."
        echo "Current value: '$EXTRA_DEP_JOB'"
        exit 1
    fi
}


submit_dialogue() {
    local job_id

    local sbatch_args=(
        --parsable
        --gres="$DIALOGUE_SBATCH_GRES"
        --export=ALL
        --gpu-bind="$SLURM_GPU_BIND"
    )

    if [[ -n "${EXTRA_DEP_JOB:-}" ]]; then
        sbatch_args+=(
            --dependency="afterok:${EXTRA_DEP_JOB}"
        )

        echo "Submitting dialogue after job ${EXTRA_DEP_JOB}..."
    else
        echo "Submitting dialogue with no external dependency..."
    fi

    job_id=$(
        sbatch \
            "${sbatch_args[@]}" \
            "$ROOT/scripts/run/sbatch_job_dialogue.sh"
    )

    echo "$job_id"
}


submit_post_after() {
    local dependency_job="$1"
    local job_id

    job_id=$(
        sbatch \
            --parsable \
            --export=ALL \
            --dependency="afterok:${dependency_job}" \
            "$ROOT/scripts/run/sbatch_job_post.sh"
    )

    echo "$job_id"
}


submit_post_only() {
    local job_id

    local sbatch_args=(
        --parsable
        --export=ALL
    )

    if [[ -n "${EXTRA_DEP_JOB:-}" ]]; then
        sbatch_args+=(
            --dependency="afterok:${EXTRA_DEP_JOB}"
        )

        echo "Submitting post after job ${EXTRA_DEP_JOB}..."
    else
        echo "Submitting post with no dependency..."
    fi

    job_id=$(
        sbatch \
            "${sbatch_args[@]}" \
            "$ROOT/scripts/run/sbatch_job_post.sh"
    )

    echo "$job_id"
}


# ============================================================
# 主逻辑
# ============================================================

check_extra_dependency

echo
echo "============================================================"
echo "Submission config"
echo "============================================================"
echo "RUN_DIALOGUE_JOB=${RUN_DIALOGUE_JOB}"
echo "EXTRA_DEP_JOB=${EXTRA_DEP_JOB:-<none>}"
echo "============================================================"
echo


case "$RUN_DIALOGUE_JOB" in

    1)
        # ----------------------------------------------------
        # dialogue -> post
        #
        # 如果 EXTRA_DEP_JOB 有值：
        #
        # EXTRA_DEP_JOB -> dialogue -> post
        #
        # 否则：
        #
        # dialogue -> post
        # ----------------------------------------------------

        JOB1="$(submit_dialogue)"

        # submit_dialogue 有提示信息，因此取最后一行作为 job id
        JOB1="$(printf '%s\n' "$JOB1" | tail -n 1)"

        if [[ -z "$JOB1" ]]; then
            echo "ERROR: failed to obtain dialogue job ID."
            exit 1
        fi

        echo "Dialogue job submitted: $JOB1"
        echo "  gres: $DIALOGUE_SBATCH_GRES"
        echo "  gpu-bind: $SLURM_GPU_BIND"

        JOB2="$(submit_post_after "$JOB1")"

        if [[ -z "$JOB2" ]]; then
            echo "ERROR: failed to obtain post job ID."
            exit 1
        fi

        echo "Post job submitted: $JOB2"
        echo "  dependency: afterok:$JOB1"

        echo
        echo "Pipeline:"
        if [[ -n "${EXTRA_DEP_JOB:-}" ]]; then
            echo "  ${EXTRA_DEP_JOB} -> ${JOB1} -> ${JOB2}"
        else
            echo "  ${JOB1} -> ${JOB2}"
        fi
        ;;


    2)
        # ----------------------------------------------------
        # dialogue only
        # ----------------------------------------------------

        JOB1="$(submit_dialogue)"
        JOB1="$(printf '%s\n' "$JOB1" | tail -n 1)"

        if [[ -z "$JOB1" ]]; then
            echo "ERROR: failed to obtain dialogue job ID."
            exit 1
        fi

        echo "Dialogue job submitted: $JOB1"
        echo "  gres: $DIALOGUE_SBATCH_GRES"
        echo "  gpu-bind: $SLURM_GPU_BIND"
        ;;


    0)
        # ----------------------------------------------------
        # post only
        # ----------------------------------------------------

        JOB2="$(submit_post_only)"
        JOB2="$(printf '%s\n' "$JOB2" | tail -n 1)"

        if [[ -z "$JOB2" ]]; then
            echo "ERROR: failed to obtain post job ID."
            exit 1
        fi

        echo "Post job submitted: $JOB2"

        if [[ -n "${EXTRA_DEP_JOB:-}" ]]; then
            echo "  dependency: afterok:${EXTRA_DEP_JOB}"
        fi
        ;;


    *)
        echo "ERROR: invalid RUN_DIALOGUE_JOB='$RUN_DIALOGUE_JOB'"
        echo
        echo "Allowed values:"
        echo "  0 = post only"
        echo "  1 = dialogue -> post"
        echo "  2 = dialogue only"
        exit 1
        ;;

esac