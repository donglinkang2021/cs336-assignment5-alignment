# todo

事情有点多，先写一下来捋清自己的思路

- attention_masks 的部分还是得加入一下，不然自己实现的版本会有点问题
    - 看一下 tokenizer 原来是可以返回 attention_masks 的，参考 [here](../myrepo/minimal-openrs/trainer/grpo_trainer.py#L1085) 来测试一下 attention_masks 到底是怎么回事
    - 实现之后看一下怎么拼在一起比较方便
    - 后面再思考一下还需不需要重新改进之前的那部分代码自己手动pad的实现 [here](cs336_alignment/sft_utils.py#L90)