# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

SWE-bench 是一个用于评估大型语言模型在真实世界软件工程问题上的基准测试工具。给定一个代码库和一个issue，模型需要生成一个能够解决该问题的补丁(patch)。

## 核心架构

项目主要由以下几个模块组成:

### 1. Harness (评估引擎) - `swebench/harness/`
- **核心功能**: 在Docker容器中运行评估,应用模型生成的补丁并运行测试
- **关键文件**:
  - `run_evaluation.py`: 评估主入口点
  - `docker_build.py`: 构建Docker镜像
  - `docker_utils.py`: Docker操作工具函数
  - `grading.py`: 测试结果评分逻辑
  - `test_spec/`: 测试规范和脚本生成

### 2. Inference (推理模块) - `swebench/inference/`
- **核心功能**: 使用不同的模型生成补丁
- **支持的模型类型**:
  - API模型 (OpenAI, Anthropic) - `run_api.py`
  - 本地Llama模型 (SWE-Llama) - `run_llama.py`
  - 实时GitHub issues - `run_live.py`
- **数据集生成**: `make_datasets/` - 生成用于训练和推理的数据集

### 3. Collect (数据收集) - `swebench/collect/`
- **核心功能**: 从GitHub仓库收集和构建评估任务实例
- **关键脚本**:
  - `print_pulls.py`: 从GitHub仓库获取PR数据
  - `build_dataset.py`: 将PR转换为任务实例
  - `get_tasks_pipeline.py`: 自动化收集流程

### 4. Versioning (版本管理) - `swebench/versioning/`
- **核心功能**: 管理和映射不同仓库的版本信息

## 常用命令

### 安装
```bash
# 从源码安装
pip install -e .

# 安装可选依赖
pip install -e ".[datasets]"      # 数据集生成和API推理
pip install -e ".[inference]"     # 本地模型推理 (需要GPU)
pip install -e ".[test]"          # 测试依赖
```

### 运行评估
```bash
# 验证安装 (单个实例)
python -m swebench.harness.run_evaluation \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --run_id validate-gold

# 在SWE-bench Lite上评估
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path <path_to_predictions> \
    --max_workers 8 \
    --run_id <run_id>

# 在完整SWE-bench上评估
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench \
    --predictions_path <path_to_predictions> \
    --max_workers 12 \
    --run_id <run_id>

# 在Modal云端运行评估
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path <path_to_predictions> \
    --parallelism 10 \
    --modal true
```

**评估注意事项**:
- 推荐在 x86_64 机器上运行,至少需要 120GB 可用存储空间、16GB RAM 和 8 CPU核心
- max_workers 建议不超过 `min(0.75 * os.cpu_count(), 24)`
- MacOS M系列或ARM系统需添加 `--namespace ''` 标志以本地构建镜像

### 运行推理
```bash
# API模型推理 (Anthropic Claude)
export ANTHROPIC_API_KEY=<your key>
python -m swebench.inference.run_api \
    --dataset_name_or_path princeton-nlp/SWE-bench_oracle \
    --model_name_or_path claude-2 \
    --output_dir ./outputs

# API模型推理 (OpenAI)
export OPENAI_API_KEY=<your key>
python -m swebench.inference.run_api \
    --dataset_name_or_path princeton-nlp/SWE-bench_oracle \
    --model_name_or_path gpt-4 \
    --output_dir ./outputs

# 本地Llama模型推理
python -m swebench.inference.run_llama \
    --dataset_path princeton-nlp/SWE-bench_oracle \
    --model_name_or_path princeton-nlp/SWE-Llama-13b \
    --output_dir ./outputs \
    --temperature 0

# 在实时GitHub issue上运行推理
python -m swebench.inference.run_live \
    --model_name gpt-3.5-turbo-1106 \
    --issue_url https://github.com/<owner>/<repo>/issues/<number>
```

### 测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_evaluation.py
pytest tests/test_harness_utils.py

# 运行带覆盖率的测试
pytest --cov=swebench
```

### Docker镜像管理
```bash
# 构建基础镜像
python -m swebench.harness.docker_build --build_base_images

# 构建环境镜像
python -m swebench.harness.docker_build --build_env_images

# 清理容器
python -m swebench.harness.remove_containers
```

## 数据格式

### 预测文件格式 (JSONL)
```json
{
  "instance_id": "repo_owner__repo_name-issue_number",
  "model_name_or_path": "model-name",
  "model_patch": "diff --git a/file.py b/file.py\n..."
}
```

### 评估结果输出
- `evaluation_results/`: 评估结果目录
  - `results.json`: 总体评估指标
  - `instance_results.jsonl`: 每个实例的详细结果
  - `run_logs/`: 每个实例的执行日志

## 开发注意事项

### Docker使用
- SWE-bench 使用Docker进行可复现的评估
- 每个任务实例在独立的Docker容器中运行
- 镜像分层: base → env → instance
- 使用 `--cache_level` 参数控制缓存策略 (none, base, env, instance)

### 模块导入
主要的API在 `swebench/__init__.py` 中导出,可以直接从 `swebench` 导入:
```python
from swebench import run_evaluation, build_dataset, get_tasks_pipeline
from swebench import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
```

### 支持的数据集
- SWE-bench (完整集): `princeton-nlp/SWE-bench`
- SWE-bench Lite (小型子集): `princeton-nlp/SWE-bench_Lite`
- SWE-bench Verified (经人工验证): `princeton-nlp/SWE-bench_Verified`
- SWE-bench Multimodal: `princeton-nlp/SWE-bench_Multimodal`

### 日志位置
- Docker构建日志: `logs/build_images/`
- 评估运行日志: `logs/run_evaluation/`

## 相关资源

- 官方文档: https://swebench.com
- GitHub Issues: https://github.com/swe-bench/SWE-bench/issues
- 论文 (ICLR 2024): https://arxiv.org/abs/2310.06770
- SWE-agent: https://github.com/SWE-agent/SWE-agent
- SWE-smith (训练数据生成): https://github.com/SWE-bench/SWE-smith
- sb-cli (云端评估): https://github.com/swe-bench/sb-cli
