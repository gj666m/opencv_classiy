#!/bin/bash
# ============================================================
# AutoDL 一键训练脚本
# 用法: bash scripts/autodl_train_all.sh
# 会依次训练 4 个模型，评估，最后打包结果
# ============================================================

set -e

echo "============================================================"
echo "  AutoDL 一键训练 — 4 个模型对比实验"
echo "============================================================"

# 安装缺失依赖
echo "[1/5] 安装依赖..."
pip install pyyaml scikit-learn tqdm -q

# 模型列表
MODELS=("mobilenet_v3_small" "resnet18" "efficientnet_b0" "shufflenet_v2")
EPOCHS=50
CONFIG="configs/train.yaml"

# 训练每个模型
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  开始训练: $MODEL ($EPOCHS epochs)"
    echo "============================================================"

    python scripts/train.py \
        --config $CONFIG \
        --model $MODEL \
        --epochs $EPOCHS

    echo "  $MODEL 训练完成!"
done

# 评估每个模型
echo ""
echo "============================================================"
echo "  开始评估所有模型"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    CKPT="checkpoints/best_${MODEL}.pth"

    if [ -f "$CKPT" ]; then
        echo ""
        echo "--- 评估: $MODEL ---"
        python scripts/evaluate.py \
            --checkpoint "$CKPT" \
            --split test
    else
        echo "  跳过 $MODEL (checkpoint 不存在)"
    fi
done

# 生成多模型对比分析
echo ""
echo "============================================================"
echo "  生成多模型对比分析"
echo "============================================================"

python scripts/compare_models.py \
    --log_dir logs \
    --report_dir outputs/reports \
    --checkpoint_dir checkpoints \
    --output_dir outputs/comparison

# 打包结果
echo ""
echo "============================================================"
echo "  打包训练结果..."
echo "============================================================"

RESULT_DIR="training_results"
rm -rf $RESULT_DIR
mkdir -p $RESULT_DIR

# 复制 checkpoints
cp checkpoints/best_*.pth $RESULT_DIR/ 2>/dev/null || true

# 复制日志和图表
cp -r logs/* $RESULT_DIR/ 2>/dev/null || true
cp -r outputs/* $RESULT_DIR/ 2>/dev/null || true

# 打包
tar -czf training_results.tar.gz $RESULT_DIR/

echo ""
echo "============================================================"
echo "  全部完成!"
echo "  结果文件: training_results.tar.gz"
echo "  内容清单:"
echo "    - checkpoints/best_*.pth          (4个模型权重)"
echo "    - train_log.csv                   (训练日志)"
echo "    - config_snapshot_*.yaml          (训练配置快照)"
echo "    - training_curve_*.png            (各模型训练曲线)"
echo "    - confusion_matrix_*.png          (混淆矩阵)"
echo "    - evaluation_report_*.md          (评估报告)"
echo "    - comparison/                     (对比分析)"
echo "        - model_comparison.md         (汇总对比表)"
echo "        - combined_curves.png         (合并曲线图)"
echo "        - comparison_bar_chart.png    (柱状对比图)"
echo "  请下载到本地后解压到项目根目录覆盖"
echo "============================================================"
