# 关于 hydra 使用 multiple configs 的一些问题

我希望可以同时加载列表中的多个配置文件，请问一般应该怎么做：比如我有  

```yaml
defaults:
- eval_dataset: competition_math
- model: gpt2
```

但是我希望可以将 eval_dataset 改成一个列表的形式（我的 eval_datasets/ 中有 competition_math.yaml, gsm8k.yaml, math500.yaml）  

```yaml
defaults:
- eval_datasets:
	- competition_math
	- gsm8k
	- math500
- model: gpt2
```

从而方便我评测，我应该怎么做：

【解决方法】很简单，只需要在 competition_math.yaml中指定成一个字典即可【下面格式】，

```bash
competition_math:
	type: huggingface
	num_samples: 100
```

不要只是指定成列表【下面格式】(如果写成这种格式导致的结果就是被最后一个数据集math500替代前面所有的值，相当于前面都没定义)

```bash
type: huggingface
num_samples: 100
```

具体例子可以参考

- https://github.com/donglinkang2021/hydra-zoo 写的一个快速验证的小demo
- https://hydra.cc/docs/patterns/select_multiple_configs_from_config_group/ hydra 官方文档例子