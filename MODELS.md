# 常用模型配置

## OpenAI 模型

### GPT-4o Mini (推荐用于测试)
```
gpt-4o-mini-2024-07-18
```

### GPT-4.1 Mini
```
gpt-4.1-mini-2025-04-14
```

### GPT-4o
```
gpt-4o-2024-11-20
```

### GPT-4 Turbo
```
gpt-4-turbo-2024-04-09
```

## Anthropic 模型

### Claude 3.5 Sonnet (最新)
```
claude-3-5-sonnet-20241022
```

### Claude 3 Opus
```
claude-3-opus-20240229
```

### Claude 3 Haiku
```
claude-3-haiku-20240307
```

## 使用方法

在运行推理时使用 `--model_name_or_path` 参数：

```bash
# Oracle 模式
python -m swebench.inference.run_api \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite_oracle \
    --model_name_or_path gpt-4o-mini-2024-07-18 \
    --output_dir ./outputs/oracle

# BM25 模式
python -m swebench.inference.run_api \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite_bm25_27K \
    --model_name_or_path gpt-4o-mini-2024-07-18 \
    --output_dir ./outputs/bm25
```
