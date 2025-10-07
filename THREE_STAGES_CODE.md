# SWE-bench 三阶段对应的代码文件

## 📋 总览

| 阶段 | 主要脚本 | 位置 | 功能 |
|------|---------|------|------|
| 阶段1: 数据准备 | `create_text_dataset.py` | `swebench/inference/make_datasets/` | 生成完整 prompt |
| 阶段2: 推理生成 | `run_api.py` / `run_llama.py` / `run_live.py` | `swebench/inference/` | 调用模型生成 patch |
| 阶段3: 评估测试 | `run_evaluation.py` | `swebench/harness/` | 在 Docker 中测试 patch |

---

## 🔷 阶段 1: 数据准备

### 核心文件结构
```
swebench/inference/make_datasets/
├── create_text_dataset.py      ← 主入口脚本
├── create_instance.py          ← prompt 格式化函数
├── bm25_retrieval.py           ← BM25 检索实现
├── tokenize_dataset.py         ← 分词器工具
└── utils.py                    ← 辅助函数
```

### 1.1 主入口: `create_text_dataset.py`

**路径**: `swebench/inference/make_datasets/create_text_dataset.py`

**关键代码**:
```python
# 第 114-254 行: main() 函数
def main(
    dataset_name_or_path,
    splits,
    file_source,          # oracle/bm25/all
    retrieval_file,
    prompt_style,         # style-2/style-3/...
    k,
    max_context_len,
    tokenizer_name,
    ...
):
    # 1. 加载原始数据集
    dataset = load_dataset(dataset_name_or_path)

    # 2. 处理每个 split
    for split in splits:
        split_instances = {x["instance_id"]: x for x in dataset[split]}

        # 3. 调用核心函数: 添加 text_inputs (prompt)
        add_text_inputs(
            split_instances,
            retrieval_file=retrieval_file,
            k=k,
            prompt_style=prompt_style,
            file_source=file_source,          # ← 决定如何获取代码
            max_context_len=max_context_len,
            tokenizer_name=tokenizer_name,
            progress_file=progress_file,
        )

    # 4. 保存新数据集
    final_dataset.save_to_disk(output_file)
```

**命令行使用**:
```bash
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite \
    --file_source bm25 \
    --retrieval_file ./bm25.jsonl \
    --k 10 \
    --prompt_style style-3 \
    --output_dir ./prepared_dataset
```

---

### 1.2 核心逻辑: `create_instance.py`

**路径**: `swebench/inference/make_datasets/create_instance.py`

#### 关键函数 1: `add_text_inputs()`
```python
# 第 340-495 行
def add_text_inputs(
    instances,
    retrieval_file,
    k,
    prompt_style,
    file_source,              # oracle/bm25/all
    max_context_len=None,
    tokenizer_name=None,
    progress_file=None,
):
    """处理实例并生成 text_inputs (完整 prompt)"""

    # 1. 如果使用 BM25,添加检索结果
    if file_source in {"bm25"}:
        add_retrieval_results(instances, retrieval_file, k, file_source)

    # 2. 对每个实例处理
    for instance_id, instance in instances.items():
        with AutoContextManager(instance, root_dir) as cm:
            # 2.1 获取 README 文件
            readmes = cm.get_readme_files()
            processed_instance["readmes"] = ingest_files(readmes)

            # 2.2 根据 file_source 获取代码文件
            if file_source == "oracle":
                # 从正确答案的 patch 中提取文件
                processed_instance["file_contents"] = ingest_files(
                    get_oracle_filenames(processed_instance)
                )
            elif file_source == "bm25":
                # 使用 BM25 检索的文件
                processed_instance["file_contents"] = ingest_files(
                    [x["docid"] for x in processed_instance["hits"]]
                )
            elif file_source == "all":
                # 包含所有文件
                processed_instance["file_contents"] = (
                    ingest_directory_contents(cm.repo_path)
                )

            # 2.3 如果有 token 限制,按 token 数筛选文件
            if max_context_len is not None:
                # 逐个添加文件,直到达到 token 上限
                for filename in files:
                    tokens = tokenizer_func(file_content, tokenizer)
                    if current_tokens + len(tokens) < max_context_len:
                        include_files.append(filename)
                        current_tokens += len(tokens)

            # 2.4 生成最终的 text_inputs (完整 prompt)
            processed_instance["text_inputs"] = PROMPT_FUNCTIONS[prompt_style](
                processed_instance
            )

            # 2.5 保存到文件
            progress_file_handle.write(json.dumps(processed_instance) + "\n")
```

#### 关键函数 2: Prompt 格式化函数

**`prompt_style_3()` - 推荐的 prompt 格式** (第 221-256 行)
```python
def prompt_style_3(instance):
    """生成 style-3 格式的 prompt"""
    premise = "You will be provided with a partial code base and an issue statement..."

    # 格式化 README
    readmes_text = make_code_text(instance["readmes"])

    # 格式化代码文件(带行号)
    code_text = make_code_text(instance["file_contents"])

    # 组装最终 prompt
    final_text = [
        premise,
        "<issue>",
        instance["problem_statement"],    # ← 问题描述
        "</issue>",
        "",
        "<code>",
        readmes_text,                     # ← README 内容
        code_text,                        # ← 相关代码文件
        "</code>",
        "",
        "Here is an example of a patch file...",
        "<patch>",
        PATCH_EXAMPLE,                    # ← patch 格式示例
        "</patch>",
        "",
        "I need you to solve the provided issue...",
        "Respond below:",
    ]

    return "\n".join(final_text)
```

**`make_code_text()` - 代码格式化** (第 127-136 行)
```python
def make_code_text(files_dict, add_line_numbers=True):
    """将代码文件格式化为带行号的文本"""
    all_text = ""
    for filename, contents in sorted(files_dict.items()):
        all_text += f"[start of {filename}]\n"

        # 添加行号
        if add_line_numbers:
            for ix, line in enumerate(contents.split("\n"), start=1):
                all_text += f"{ix} {line}\n"
        else:
            all_text += contents

        all_text += f"[end of {filename}]\n"

    return all_text.strip("\n")
```

**`get_oracle_filenames()` - Oracle 模式** (第 326-337 行)
```python
def get_oracle_filenames(instance):
    """从正确的 patch 中提取被修改的文件名"""
    source_files = {
        patch_file.source_file.split("a/", 1)[-1]
        for patch_file in unidiff.PatchSet(instance["patch"])
    }
    return source_files
```

**其他 prompt 样式**:
- `prompt_style_2()`: 第 165-190 行
- `full_file_gen()`: 第 259-284 行
- `style_2_edits_only()`: 第 193-218 行

**PROMPT_FUNCTIONS 字典** (第 296-301 行):
```python
PROMPT_FUNCTIONS = {
    "style-2": prompt_style_2,
    "style-3": prompt_style_3,
    "full_file_gen": full_file_gen,
    "style-2-edits-only": prompt_style_2_edits_only,
}
```

---

### 1.3 BM25 检索: `bm25_retrieval.py`

**路径**: `swebench/inference/make_datasets/bm25_retrieval.py`

**关键函数**:
```python
# make_index(): 构建 BM25 索引
def make_index(repo_dir, query, commit, document_encoding_func, ...):
    """为仓库构建 BM25 索引"""
    # 1. 获取所有代码文件
    # 2. 编码为文档
    # 3. 构建 Pyserini 索引
    # 4. 返回索引路径

# search(): 检索相关文件
def search(instance, index_dir):
    """使用 BM25 检索最相关的文件"""
    # 1. 加载索引
    # 2. 用 problem_statement 作为查询
    # 3. 返回 Top-K 文件
```

**命令行使用**:
```bash
python -m swebench.inference.make_datasets.bm25_retrieval \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --output_file ./bm25_results.jsonl
```

---

### 1.4 分词工具: `tokenize_dataset.py`

**路径**: `swebench/inference/make_datasets/tokenize_dataset.py`

**关键代码** (第 31-34 行):
```python
TOKENIZER_FUNCS = {
    "cl100k": (tiktoken.get_encoding("cl100k_base"), cl100k),
    "llama": (LlamaTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K"), llama),
}
```

---

## 🔷 阶段 2: 推理生成

### 核心文件结构
```
swebench/inference/
├── run_api.py          ← API 模型推理 (GPT/Claude)
├── run_llama.py        ← 本地 LLaMA 模型推理
└── run_live.py         ← 实时 GitHub issue 推理
```

### 2.1 API 推理: `run_api.py`

**路径**: `swebench/inference/run_api.py`

**关键代码**:

#### main() 函数 (第 442-508 行)
```python
def main(
    dataset_name_or_path,
    split,
    model_name_or_path,    # gpt-4, claude-2, etc.
    output_dir,
    model_args,
    max_cost,
):
    # 1. 加载准备好的数据集 (阶段1的输出)
    dataset = load_dataset(dataset_name_or_path)[split]

    # 2. 按输入长度排序
    lens = np.array(list(map(len, dataset["text"])))
    dataset = dataset.select(np.argsort(lens))

    # 3. 调用推理函数
    if model_name_or_path.startswith("claude"):
        anthropic_inference(dataset, model_name_or_path, output_file, ...)
    elif model_name_or_path.startswith("gpt"):
        openai_inference(dataset, model_name_or_path, output_file, ...)
```

#### OpenAI 推理 (第 174-243 行)
```python
def openai_inference(test_dataset, model_name_or_path, output_file, ...):
    """使用 OpenAI API 运行推理"""

    # 1. 设置 API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # 2. 对每个实例推理
    for datum in tqdm(test_dataset):
        instance_id = datum["instance_id"]

        # 3. 读取准备好的 prompt
        text_prompt = f"{datum['text']}\n\n"

        # 4. 调用 OpenAI API
        response, cost = call_chat(
            model_name_or_path,
            text_prompt,              # ← 使用阶段1生成的 prompt
            use_azure=False,
            temperature=0.2,
            top_p=0.95,
        )

        # 5. 提取生成的内容
        completion = response.choices[0].message.content

        # 6. 从生成内容中提取 patch
        model_patch = extract_diff(completion)

        # 7. 保存结果
        output_dict = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "full_output": completion,
            "model_patch": model_patch,    # ← 提取的 patch
        }
        print(json.dumps(output_dict), file=f, flush=True)
```

#### call_chat() - OpenAI API 调用 (第 114-159 行)
```python
@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
def call_chat(model_name_or_path, inputs, use_azure, temperature, top_p, **model_args):
    """调用 OpenAI API"""

    # 1. 分割 system message 和 user message
    system_messages = inputs.split("\n", 1)[0]
    user_message = inputs.split("\n", 1)[1]

    # 2. 调用 API
    response = openai.chat.completions.create(
        model=model_name_or_path,
        messages=[
            {"role": "system", "content": system_messages},
            {"role": "user", "content": user_message},    # ← 发送 prompt
        ],
        temperature=temperature,
        top_p=top_p,
        **model_args,
    )

    # 3. 计算费用
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calc_cost(response.model, input_tokens, output_tokens)

    return response, cost
```

#### Anthropic 推理 (第 323-403 行)
```python
def anthropic_inference(test_dataset, model_name_or_path, output_file, ...):
    """使用 Anthropic API 运行推理"""

    # 类似 OpenAI,但使用 Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    anthropic = Anthropic(api_key=api_key)

    for datum in tqdm(test_dataset):
        # 使用 datum["text"] 作为 prompt
        text_inputs = f"{datum['text']}\n"

        # 调用 Anthropic API
        completion, cost = call_anthropic(
            text_inputs,
            anthropic,
            model_name_or_path,
            temperature=0.2,
            top_p=0.95,
        )

        # 提取 patch 并保存
        ...
```

**命令行使用**:
```bash
# OpenAI
export OPENAI_API_KEY="sk-xxx"
python -m swebench.inference.run_api \
    --dataset_name_or_path ./prepared_dataset \
    --model_name_or_path gpt-4 \
    --output_dir ./predictions

# Anthropic
export ANTHROPIC_API_KEY="sk-xxx"
python -m swebench.inference.run_api \
    --dataset_name_or_path ./prepared_dataset \
    --model_name_or_path claude-3-opus-20240229 \
    --output_dir ./predictions
```

---

### 2.2 本地 LLaMA 推理: `run_llama.py`

**路径**: `swebench/inference/run_llama.py`

**关键代码**:

#### main() 函数 (第 360-424 行)
```python
def main(
    model_name_or_path,
    peft_path,
    dataset_path,
    split,
    temperature,
    top_p,
    output_dir,
    ...
):
    # 1. 加载模型
    model = load_model(model_name_or_path, peft_path)
    tokenizer = load_tokenizer(model_name_or_path)

    # 2. 加载数据
    dataset = load_data(dataset_path, split, tokenizer, ...)

    # 3. 生成
    with open(output_file, "a") as f:
        generate(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            temperature=temperature,
            top_p=top_p,
            fileobj=f,
            ...
        )
```

#### generate() - 生成函数 (第 246-335 行)
```python
def generate(model, dataset, tokenizer, temperature, top_p, fileobj, ...):
    """使用 LLaMA 模型生成 patch"""

    for instance in tqdm(dataset):
        # 1. 准备 input_ids (已经在 load_data 时 tokenize)
        input_ids = torch.tensor([instance["input_ids"]], device=model.device)

        # 2. 生成
        output = model.generate(
            input_ids=input_ids,
            temperature=1.0 if temperature == 0 else temperature,
            top_p=top_p,
            do_sample=False if temperature == 0 else True,
            max_new_tokens=200,
            stopping_criteria=stopping_criteria,
        )

        # 3. 解码
        output_text = tokenizer.decode(output[0], skip_special_tokens=False)

        # 4. 提取 diff
        diff = extract_diff(output_text)

        # 5. 保存
        result = {
            "instance_id": instance["instance_id"],
            "full_output": output_text,
            "model_patch": diff,
            "model_name_or_path": model_name_or_path,
        }
        print(json.dumps(result), file=fileobj, flush=True)
```

**命令行使用**:
```bash
python -m swebench.inference.run_llama \
    --dataset_path ./prepared_dataset \
    --model_name_or_path princeton-nlp/SWE-Llama-13b \
    --output_dir ./predictions \
    --temperature 0
```

---

### 2.3 实时推理: `run_live.py`

**路径**: `swebench/inference/run_live.py`

**关键代码** (第 173-260 行):
```python
def main(model_name, prompt_style, issue_url, ...):
    """对实时 GitHub issue 运行推理"""

    # 1. 解析 issue URL
    owner, repo, issue_num = parse_issue_url(issue_url)

    # 2. 获取 issue 内容
    problem_statement = get_problem_statement(owner, repo, issue_num, gh)

    # 3. 创建实例 (包含检索和 prompt 生成)
    instance = make_instance(
        owner, repo, problem_statement, commit, root_dir,
        tokenizer, prompt_style, max_context_len, ...
    )

    # 4. 调用模型
    if model_name.startswith("gpt"):
        response, _ = call_chat(model_name, instance["text_inputs"], ...)
        completion = response.choices[0].message.content
    else:
        # Anthropic
        completion = call_anthropic(instance["text_inputs"], ...)

    # 5. 提取 patch 并保存
    model_patch = extract_diff(completion)
```

---

## 🔷 阶段 3: 评估测试

### 核心文件结构
```
swebench/harness/
├── run_evaluation.py       ← 评估主脚本
├── docker_build.py         ← Docker 镜像构建
├── docker_utils.py         ← Docker 操作工具
├── grading.py              ← 评分逻辑
└── test_spec/              ← 测试规范生成
    ├── test_spec.py
    └── create_scripts.py
```

### 3.1 评估主脚本: `run_evaluation.py`

**路径**: `swebench/harness/run_evaluation.py`

**关键代码**:

#### main() 函数 (简化版逻辑)
```python
def main(
    dataset_name,
    predictions_path,    # 阶段2生成的 JSONL 文件
    max_workers,
    run_id,
    ...
):
    # 1. 加载数据集
    dataset = load_dataset(dataset_name)

    # 2. 加载模型预测
    predictions = load_predictions(predictions_path)

    # 3. 对每个预测运行评估
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for prediction in predictions:
            future = executor.submit(
                run_instance_evaluation,
                prediction,
                dataset,
                run_id,
            )
            futures.append(future)

        # 4. 收集结果
        results = [f.result() for f in futures]

    # 5. 生成报告
    generate_report(results, run_id)
```

#### run_instance_evaluation() - 单实例评估
```python
def run_instance_evaluation(prediction, dataset, run_id):
    """评估单个实例"""

    # 1. 获取对应的数据集实例
    instance = get_instance(dataset, prediction["instance_id"])

    # 2. 构建 Docker 容器
    container = build_container(instance)

    # 3. 在容器中应用 patch
    apply_result = apply_patch_in_container(
        container,
        prediction["model_patch"]    # ← 模型生成的 patch
    )

    # 4. 运行测试
    if apply_result.success:
        test_result = run_tests_in_container(
            container,
            instance["test_patch"],
            instance["FAIL_TO_PASS"],
            instance["PASS_TO_PASS"],
        )
    else:
        test_result = {"resolved": False, "error": "patch failed to apply"}

    # 5. 判断是否解决
    resolved = (
        apply_result.success and
        all_fail_to_pass_tests_passed(test_result) and
        all_pass_to_pass_tests_passed(test_result)
    )

    # 6. 返回结果
    return {
        "instance_id": prediction["instance_id"],
        "resolved": resolved,
        "test_result": test_result,
        "apply_result": apply_result,
    }
```

---

### 3.2 评分逻辑: `grading.py`

**路径**: `swebench/harness/grading.py`

**关键函数**:
```python
def get_eval_report(
    test_spec,
    prediction,
    log_path,
    include_tests_status=False,
):
    """生成评估报告"""

    # 1. 解析测试日志
    log_content = open(log_path).read()

    # 2. 提取测试结果
    fail_to_pass_results = parse_log_for_tests(
        log_content,
        test_spec["FAIL_TO_PASS"]
    )
    pass_to_pass_results = parse_log_for_tests(
        log_content,
        test_spec["PASS_TO_PASS"]
    )

    # 3. 判断是否解决
    resolved = (
        all(result == "PASSED" for result in fail_to_pass_results) and
        all(result == "PASSED" for result in pass_to_pass_results)
    )

    return {
        "instance_id": test_spec["instance_id"],
        "resolved": resolved,
        "fail_to_pass": fail_to_pass_results,
        "pass_to_pass": pass_to_pass_results,
    }
```

---

### 3.3 Docker 工具: `docker_utils.py`

**路径**: `swebench/harness/docker_utils.py`

**关键函数**:
```python
def run_docker_container(container_name, image_name, timeout=3600):
    """运行 Docker 容器"""
    client = docker.from_env()
    container = client.containers.run(
        image_name,
        name=container_name,
        detach=True,
        ...
    )
    return container

def exec_run_with_timeout(container, command, timeout=3600):
    """在容器中执行命令"""
    exit_code, output = container.exec_run(
        command,
        workdir="/testbed",
    )
    return exit_code, output
```

**命令行使用**:
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path ./predictions/gpt-4__predictions.jsonl \
    --max_workers 8 \
    --run_id gpt4_eval
```

---

## 📊 完整代码调用链

### 阶段 1 调用链
```
create_text_dataset.py:main()
    ↓
create_text_dataset.py:add_text_inputs()
    ↓
create_instance.py:add_text_inputs()
    ├─→ bm25_retrieval.py:search()            [如果 file_source=bm25]
    ├─→ create_instance.py:get_oracle_filenames() [如果 file_source=oracle]
    ├─→ utils.py:ingest_directory_contents()   [如果 file_source=all]
    ├─→ tokenize_dataset.py:TOKENIZER_FUNCS   [如果有 max_context_len]
    └─→ create_instance.py:PROMPT_FUNCTIONS[prompt_style]()
        ├─→ prompt_style_3()
        └─→ make_code_text()
```

### 阶段 2 调用链 (API)
```
run_api.py:main()
    ↓
run_api.py:openai_inference() 或 anthropic_inference()
    ↓
run_api.py:call_chat() 或 call_anthropic()
    ↓
OpenAI/Anthropic API
    ↓
utils.py:extract_diff()
```

### 阶段 2 调用链 (LLaMA)
```
run_llama.py:main()
    ↓
run_llama.py:load_model()
    ↓
run_llama.py:load_data()
    ↓
run_llama.py:generate()
    ↓
utils.py:extract_diff()
```

### 阶段 3 调用链
```
run_evaluation.py:main()
    ↓
run_evaluation.py:run_instance_evaluation() [并行执行]
    ↓
docker_build.py:build_container()
    ↓
docker_utils.py:run_docker_container()
    ↓
docker_utils.py:exec_run_with_timeout() [应用 patch]
    ↓
docker_utils.py:exec_run_with_timeout() [运行测试]
    ↓
grading.py:get_eval_report()
```

---

## 🎯 快速查找代码的技巧

### 方法 1: 按功能查找
```bash
# 查找 prompt 相关
grep -r "def prompt_style" swebench/inference/make_datasets/

# 查找 API 调用
grep -r "openai.chat.completions" swebench/inference/

# 查找评估逻辑
grep -r "def get_eval_report" swebench/harness/
```

### 方法 2: 按文件结构
```
阶段1: swebench/inference/make_datasets/*.py
阶段2: swebench/inference/run_*.py
阶段3: swebench/harness/*.py
```

### 方法 3: 从命令行入口追踪
```bash
# 找到命令对应的文件
python -m swebench.inference.make_datasets.create_text_dataset
# → swebench/inference/make_datasets/create_text_dataset.py

python -m swebench.inference.run_api
# → swebench/inference/run_api.py

python -m swebench.harness.run_evaluation
# → swebench/harness/run_evaluation.py
```

---

**创建时间**: 2025-10-06
**维护者**: SWE-bench Team
