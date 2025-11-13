# todo

事情有点多，先写一下来捋清自己的思路

- 【better sft script】还是打算先完善一下可打印的metric，方便后面去eval
- 测试一下sft的模型的效果的如何，打算用vllm和hf的推理同时测一下
- 但是如果要执行测试记得把 rope_theta 给改回来，否则计算的值不match了
- 为了实现长上下文扩展这里将 Qwen2.5-Math-1.5B 的 config.json
    - rope_theta 10000 -> 160000
    - max_position_embeddings 4096 -> 32768
- 最后是采用trl的做法将vllm包装成server和client的形式来同步权重
- 还是在实现 vllm 的实现上踩坑了，可能还是得用下 ray 来做分布式
    - 参考这个 [here](../myrepo/minimal-openrs/vllm_demo.py)，这个是使用 trl 的 trl.extras.vllm_client
    - 或者是参考 这个 [here](../myrepo/Zero1/docs/vllm_rlhf.md)，这个是OpenRLHF 的做法
- 实现sft的主要功能要参考的地方
    - 参考这个来单独打印trained model生成的效果 [here](scripts/test_log_generations.py)
    - 参考这个的eval效果 [here](scripts/evaluate_math_baseline.py)
    - 利用这里的函数 [here](cs336_alignment/sft_utils.py)
    - 参考文档 [here](docs/alignment/4-sft/4.3-sft-experiment.md)
- attention_masks 的部分还是得加入一下，不然自己实现的版本会有点问题
    - 看一下 tokenizer 原来是可以返回 attention_masks 的，参考 [here](../myrepo/minimal-openrs/trainer/grpo_trainer.py#L1085) 来测试一下 attention_masks 到底是怎么回事
    - 实现之后看一下怎么拼在一起比较方便
    - 后面再思考一下还需不需要重新改进之前的那部分代码自己手动pad的实现 [here](cs336_alignment/sft_utils.py#L90)