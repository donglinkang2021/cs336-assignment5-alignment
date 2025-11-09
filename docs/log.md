# Log

## [0.0.3] - 2025.11.09

- pass `uv run pytest -k test_tokenize_prompt_and_output` in 0.52s

## [0.0.2] - 2025.11.06

- change the tests/conftest.py to use models/Qwen2.5-Math-1.5B instead of /data/a5-alignment/models/Qwen2.5-Math-1.5
- 用 competition_math 数据集创建自定义数据集，生成 12k 训练集和 500 验证集，使用随机种子 42；
- 评测了Qwen2.5-Math-1.5B和Qwen2.5-Math-1.5B-Instruct的 zero-shot 模型效果，发现了base模型基本就可以回答对大部分的competition_math的问题，只是格式不对的问题导致没有检测出正确的答案；
- 写了一个可以 visualize .jsol数据集的前端代码；

## [0.0.1] - 2025.11.05

- 使用 modelscope 下载模型
- 更新完 主要作业部分的 markdown 文档
