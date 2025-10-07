# SWE-bench 数据集生成参数详解

本文档详细介绍 `create_text_dataset.py` 的所有参数及其使用方法。

---

## 📋 参数总览

```bash
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path <路径或名称> \
    --splits train test \
    --validation_ratio 0.01 \
    --output_dir <输出目录> \
    --retrieval_file <检索结果文件> \
    --prompt_style style-3 \
    --file_source bm25 \
    --k 10 \
    --max_context_len 16000 \
    --tokenizer_name cl100k \
    --push_to_hub_user <HuggingFace用户名>
```

---

## 1️⃣ `--dataset_name_or_path`

### 📖 说明
指定输入数据集的来源

### 🎯 类型
`str`

### 💡 默认值
`"SWE-bench/SWE-bench"`

### 📝 用途
- **HuggingFace 数据集**: 使用数据集名称,如 `princeton-nlp/SWE-bench`
- **本地数据集**: 使用 `save_to_disk()` 保存的目录路径

### 💻 示例
```bash
# 使用 HuggingFace 数据集
--dataset_name_or_path princeton-nlp/SWE-bench

# 使用本地数据集
--dataset_name_or_path /path/to/saved/dataset

# 使用 SWE-bench Lite
--dataset_name_or_path princeton-nlp/SWE-bench_Lite
```

### 📌 注意
- 数据集必须包含以下字段:
  - `instance_id`: 任务唯一标识
  - `problem_statement`: 问题描述
  - `patch`: 正确的补丁
  - `repo`: 仓库名称
  - `base_commit`: 基础 commit SHA

---

## 2️⃣ `--splits`

### 📖 说明
指定要处理的数据集分割(split)

### 🎯 类型
`list[str]` (可以指定多个)

### 💡 默认值
`["train", "test"]`

### 📝 用途
选择数据集中的哪些分割进行处理

### 💻 示例
```bash
# 只处理测试集
--splits test

# 处理训练集和测试集
--splits train test

# 处理所有分割
--splits train test dev
```

### 📌 常见分割
- `train`: 训练集
- `test`: 测试集
- `dev`: 开发集(如果存在)
- `validation`: 验证集(如果存在)

---

## 3️⃣ `--validation_ratio`

### 📖 说明
从训练集中划分出验证集的比例

### 🎯 类型
`float`

### 💡 默认值
`0.01` (1%)

### 📝 用途
自动从训练集中分割出一部分作为验证集

### 💻 示例
```bash
# 使用 1% 的训练数据作为验证集
--validation_ratio 0.01

# 使用 5% 的训练数据作为验证集
--validation_ratio 0.05

# 不创建验证集
--validation_ratio 0
```

### 📌 行为
- 只对 `train` 分割生效
- 使用 `train_test_split()` 方法,随机种子为 42
- 生成后的数据集会包含 `validation` 分割

---

## 4️⃣ `--output_dir`

### 📖 说明
保存生成的数据集的目录

### 🎯 类型
`str` (必需参数,除非使用 `--push_to_hub_user`)

### 💡 默认值
无

### 📝 用途
指定数据集的保存位置

### 💻 示例
```bash
# 保存到本地目录
--output_dir ./processed_datasets

# 保存到绝对路径
--output_dir /data/swebench/oracle_dataset
```

### 📌 输出文件命名
自动生成文件名格式:
```
{dataset_name}__{prompt_style}__fs-{file_source}[__k-{k}][__mcc-{max_context_len}-{tokenizer_name}]
```

示例:
```
SWE-bench__style-3__fs-bm25__k-10__mcc-16000-cl100k/
├── dataset_dict.json
├── test/
└── train/
```

---

## 5️⃣ `--retrieval_file`

### 📖 说明
BM25 检索结果文件的路径

### 🎯 类型
`str`

### 💡 默认值
`None`

### 📝 用途
当使用 `--file_source bm25` 时,提供预先计算的检索结果

### 💻 示例
```bash
# 先运行 BM25 检索
python -m swebench.inference.make_datasets.bm25_retrieval \
    --dataset_name princeton-nlp/SWE-bench \
    --output_file ./bm25_results.jsonl

# 使用检索结果
--retrieval_file ./bm25_results.jsonl
```

### 📌 文件格式
JSONL 格式,每行包含:
```json
{
  "instance_id": "django__django-12345",
  "hits": [
    {"docid": "path/to/file1.py", "score": 15.2},
    {"docid": "path/to/file2.py", "score": 12.8},
    ...
  ]
}
```

### ⚠️ 注意
- 仅当 `--file_source bm25` 时需要
- 使用 `oracle` 或 `all` 时忽略此参数

---

## 6️⃣ `--prompt_style`

### 📖 说明
选择 prompt 的格式样式

### 🎯 类型
`str`

### 💡 默认值
`"style-3"`

### ✅ 可选值
- `style-2`
- `style-3` (推荐)
- `full_file_gen`
- `style-2-edits-only`

### 📝 各样式对比

#### **style-2**
```
You will be provided with a partial code base and an issue statement...

<issue>
{problem_statement}
</issue>

<code>
{完整代码文件,带行号}
</code>

I need you to solve this issue by generating a single patch file...

<patch>
{示例 patch}
</patch>
```

#### **style-3** (推荐)
```
You will be provided with a partial code base and an issue statement...

<issue>
{problem_statement}
</issue>

<code>
{完整代码文件,带行号}
</code>

Here is an example of a patch file. It consists of changes to the code base...

<patch>
{示例 patch}
</patch>

I need you to solve the provided issue by generating a single patch file...
Respond below:
```

**区别**: style-3 添加了对 patch 格式的解释和明确的响应提示

#### **full_file_gen**
```
You will be provided with a partial code base and an issue statement...

<issue>
{problem_statement}
</issue>

<code>
{完整代码文件,不带行号}
</code>

I need you to solve this issue by regenerating the full files...

<example>
{完整文件示例}
</example>
```

**特点**:
- 要求模型生成完整文件而非 patch
- 代码不带行号
- 适合某些特定场景

#### **style-2-edits-only**
```
{与 style-2 相同,但只显示需要编辑的代码片段}

<code>
[start of file.py]
...
10 def function():
11     old_code
12     return value
...
[end of file.py]
</code>
```

**特点**:
- 只显示需要修改的代码区域(前后各 15 行)
- 节省 token
- 需要配合 oracle 使用(因为需要知道修改位置)

### 💻 示例
```bash
# 使用推荐的 style-3
--prompt_style style-3

# 使用 full_file_gen 生成完整文件
--prompt_style full_file_gen
```

---

## 7️⃣ `--file_source`

### 📖 说明
选择如何获取相关代码文件

### 🎯 类型
`str`

### 💡 默认值
`"oracle"`

### ✅ 可选值
- `oracle`: 从正确答案中提取文件
- `bm25`: 使用 BM25 检索
- `all`: 包含所有文件

### 📝 详细说明

#### **oracle** (预言机模式)
```python
# 从正确的 patch 中提取被修改的文件
patch = unidiff.PatchSet(instance["patch"])
files = {pf.source_file for pf in patch}
```

**优点**:
- ✅ 提供最精确的代码上下文
- ✅ 确保包含所有必需文件
- ✅ 适合研究模型的理论上界

**缺点**:
- ❌ 需要知道正确答案
- ❌ 实际应用中不可用

**用途**:
- 训练数据生成
- 性能上界测试
- 对比实验基准

#### **bm25** (检索模式)
```python
# 使用 BM25 算法检索最相关的文件
hits = search(problem_statement, bm25_index)
files = [hit["docid"] for hit in hits[:k]]
```

**优点**:
- ✅ 不需要知道答案
- ✅ 模拟真实场景
- ✅ 可配置检索数量

**缺点**:
- ❌ 可能检索到不相关文件
- ❌ 可能遗漏关键文件

**用途**:
- 实际推理
- 真实场景评估

**需要配合**:
- `--retrieval_file`: BM25 检索结果
- `--k`: Top-K 文件数量
- `--max_context_len`: token 限制

#### **all** (全量模式)
```python
# 包含仓库中所有代码文件
files = ingest_directory_contents(repo_path)
```

**优点**:
- ✅ 包含所有信息
- ✅ 不会遗漏文件

**缺点**:
- ❌ token 消耗巨大
- ❌ 大多数文件无关
- ❌ 超过模型上下文限制

**用途**:
- 测试长上下文模型
- 特殊研究场景

### 💻 示例
```bash
# Oracle 模式(研究用)
--file_source oracle

# BM25 模式(实际应用)
--file_source bm25 \
--retrieval_file ./bm25_results.jsonl \
--k 10 \
--max_context_len 16000

# 全量模式(测试长上下文)
--file_source all
```

---

## 8️⃣ `--k`

### 📖 说明
使用 BM25 检索时,最多包含多少个文件

### 🎯 类型
`int`

### 💡 默认值
`None`

### 📝 用途
限制检索文件的数量,避免 token 消耗过大

### 💻 示例
```bash
# 包含 Top-10 最相关的文件
--k 10

# 包含 Top-20 最相关的文件
--k 20

# 不限制数量(受 max_context_len 约束)
# (不指定 --k)
```

### 📌 工作原理
```python
# 按相关性排序,取前 k 个
hits = retrieval_results[:k]
for hit in hits:
    file_contents[hit["docid"]] = read_file(hit["docid"])
```

### ⚠️ 注意
- 仅当 `--file_source bm25` 时生效
- 如果同时指定 `--max_context_len`,会在 token 限制和文件数量限制中取较小值
- 使用 `oracle` 或 `all` 时忽略此参数

---

## 9️⃣ `--max_context_len`

### 📖 说明
限制 prompt 的最大 token 数量

### 🎯 类型
`int`

### 💡 默认值
`None` (不限制)

### 📝 用途
确保生成的 prompt 不超过模型的上下文窗口

### 💻 示例
```bash
# GPT-3.5 (4K 上下文)
--max_context_len 4000

# GPT-4 (8K 上下文)
--max_context_len 8000

# GPT-4-32K
--max_context_len 32000

# Claude-2 (100K 上下文)
--max_context_len 100000
```

### 📌 工作原理
```python
# 1. 计算基础 prompt 的 token 数
base_tokens = tokenizer(base_prompt)

# 2. 逐个添加文件,直到达到上限
current_tokens = base_tokens
for file in retrieved_files:
    file_tokens = tokenizer(file_content)
    if current_tokens + file_tokens < max_context_len:
        include_files.append(file)
        current_tokens += file_tokens
    else:
        break  # 达到上限,停止添加
```

### 📊 常用配置

| 模型 | 上下文窗口 | 推荐设置 | 备注 |
|------|-----------|---------|------|
| GPT-3.5-turbo | 4K | 3500 | 留出 500 给输出 |
| GPT-3.5-turbo-16k | 16K | 15000 | 留出 1000 给输出 |
| GPT-4 | 8K | 7500 | 留出 500 给输出 |
| GPT-4-32K | 32K | 30000 | 留出 2000 给输出 |
| GPT-4-turbo | 128K | 120000 | 留出 8000 给输出 |
| Claude-2 | 100K | 95000 | 留出 5000 给输出 |
| Claude-3 | 200K | 190000 | 留出 10000 给输出 |

### ⚠️ 注意
- 必须配合 `--tokenizer_name` 使用
- 不能与 `--file_source oracle` 或 `all` 同时使用
- 只对 `bm25` 模式有效

---

## 🔟 `--tokenizer_name`

### 📖 说明
指定用于计算 token 数量的分词器

### 🎯 类型
`str`

### 💡 默认值
`None`

### ✅ 可选值
- `cl100k`: OpenAI GPT-3.5/GPT-4 使用的分词器
- `llama`: LLaMA 系列模型的分词器

### 📝 详细说明

#### **cl100k**
```python
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode(text)
```

**适用模型**:
- GPT-3.5-turbo 系列
- GPT-4 系列
- text-embedding-ada-002

**特点**:
- 词汇表大小: 100,000
- 高效的多语言支持
- 不支持多进程(自动使用单进程)

#### **llama**
```python
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
tokens = tokenizer(text)["input_ids"]
```

**适用模型**:
- LLaMA 系列
- SWE-Llama
- LLaMA-2

**特点**:
- 词汇表大小: 32,000
- 支持多进程
- 需要 GPU 环境

### 💻 示例
```bash
# 使用 GPT-4
--tokenizer_name cl100k \
--max_context_len 8000

# 使用 LLaMA
--tokenizer_name llama \
--max_context_len 16000
```

### ⚠️ 注意
- 仅当指定 `--max_context_len` 时需要
- 选择与目标模型匹配的分词器
- cl100k 不支持多进程,会自动降为单进程

---

## 1️⃣1️⃣ `--push_to_hub_user`

### 📖 说明
将生成的数据集推送到 HuggingFace Hub

### 🎯 类型
`str`

### 💡 默认值
`None` (保存到本地)

### 📝 用途
直接将数据集上传到 HuggingFace,便于分享和使用

### 💻 示例
```bash
# 推送到 HuggingFace Hub
export HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxx"

python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench \
    --file_source bm25 \
    --retrieval_file ./bm25_results.jsonl \
    --k 10 \
    --prompt_style style-3 \
    --push_to_hub_user your-username

# 生成的数据集会上传到:
# https://huggingface.co/datasets/your-username/SWE-bench__style-3__fs-bm25__k-10
```

### 📌 前置条件
```bash
# 1. 安装 HuggingFace Hub
pip install huggingface-hub

# 2. 登录 HuggingFace
huggingface-cli login

# 或者设置环境变量
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

### ⚠️ 注意
- 必须设置 `HUGGING_FACE_HUB_TOKEN` 环境变量
- 不能同时指定 `--output_dir`
- 数据集会自动命名

---

## 📊 完整使用示例

### 示例 1: 创建 Oracle 数据集(研究用)
```bash
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite \
    --splits test \
    --file_source oracle \
    --prompt_style style-3 \
    --output_dir ./datasets/oracle
```

### 示例 2: 创建 BM25 检索数据集(实际应用)
```bash
# 步骤 1: 运行 BM25 检索
python -m swebench.inference.make_datasets.bm25_retrieval \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --output_file ./bm25_lite.jsonl

# 步骤 2: 创建数据集
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite \
    --splits test \
    --file_source bm25 \
    --retrieval_file ./bm25_lite.jsonl \
    --k 10 \
    --max_context_len 16000 \
    --tokenizer_name cl100k \
    --prompt_style style-3 \
    --output_dir ./datasets/bm25_16k
```

### 示例 3: 创建训练数据集并推送到 Hub
```bash
export HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxx"

python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench \
    --splits train test \
    --validation_ratio 0.05 \
    --file_source oracle \
    --prompt_style style-3 \
    --push_to_hub_user my-username
```

### 示例 4: 多种配置对比
```bash
# 配置 A: Oracle + style-3
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite \
    --file_source oracle \
    --prompt_style style-3 \
    --output_dir ./configs/oracle_style3

# 配置 B: BM25 (k=5) + style-2
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite \
    --file_source bm25 \
    --retrieval_file ./bm25_lite.jsonl \
    --k 5 \
    --prompt_style style-2 \
    --output_dir ./configs/bm25_k5_style2

# 配置 C: BM25 (token限制) + full_file_gen
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite \
    --file_source bm25 \
    --retrieval_file ./bm25_lite.jsonl \
    --max_context_len 32000 \
    --tokenizer_name cl100k \
    --prompt_style full_file_gen \
    --output_dir ./configs/bm25_32k_fullgen
```

---

## 🎯 参数组合建议

### 场景 1: 快速测试
```bash
--dataset_name_or_path princeton-nlp/SWE-bench_Lite \
--splits test \
--file_source oracle \
--prompt_style style-3 \
--output_dir ./quick_test
```

### 场景 2: 训练模型
```bash
--dataset_name_or_path princeton-nlp/SWE-bench \
--splits train \
--validation_ratio 0.05 \
--file_source oracle \
--prompt_style style-3 \
--output_dir ./training_data
```

### 场景 3: 真实推理评估
```bash
--dataset_name_or_path princeton-nlp/SWE-bench \
--splits test \
--file_source bm25 \
--retrieval_file ./bm25_results.jsonl \
--k 10 \
--max_context_len 16000 \
--tokenizer_name cl100k \
--prompt_style style-3 \
--output_dir ./eval_data
```

### 场景 4: 长上下文模型
```bash
--dataset_name_or_path princeton-nlp/SWE-bench \
--splits test \
--file_source bm25 \
--retrieval_file ./bm25_results.jsonl \
--max_context_len 100000 \
--tokenizer_name cl100k \
--prompt_style style-3 \
--output_dir ./long_context_data
```

---

## ⚠️ 常见错误和解决方案

### 错误 1: `Cannot use max_context_len with oracle`
```bash
# ❌ 错误
--file_source oracle --max_context_len 16000

# ✅ 正确
--file_source bm25 --max_context_len 16000
```

### 错误 2: `Must specify tokenizer_name if using max_context_len`
```bash
# ❌ 错误
--max_context_len 16000

# ✅ 正确
--max_context_len 16000 --tokenizer_name cl100k
```

### 错误 3: `retrieval_file not found`
```bash
# 先生成检索结果
python -m swebench.inference.make_datasets.bm25_retrieval \
    --dataset_name princeton-nlp/SWE-bench \
    --output_file ./bm25_results.jsonl

# 然后使用
--retrieval_file ./bm25_results.jsonl
```

### 错误 4: `Cannot provide output_dir if pushing to the Hub`
```bash
# ❌ 错误
--push_to_hub_user my-user --output_dir ./data

# ✅ 正确(二选一)
--push_to_hub_user my-user  # 推送到 Hub
--output_dir ./data         # 保存到本地
```

---

## 📚 参考资源

- [SWE-bench 官方文档](https://swebench.com)
- [HuggingFace Datasets 文档](https://huggingface.co/docs/datasets)
- [Tiktoken 文档](https://github.com/openai/tiktoken)
- [BM25 检索文档](./bm25_retrieval.md)

---

**最后更新**: 2025-10-06
**维护者**: SWE-bench Team
