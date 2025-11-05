# 2 Reasoning with Language Models

## 2.1 Motivation

One of the remarkable use cases of language models is in building generalist systems that can handle a wide range of natural language processing tasks. In this assignment, we will focus on a developing use case for language models: mathematical reasoning. It will serve as a testbed for us to set up evaluations, perform supervised finetuning, and experiment with teaching LMs to reason using reinforcement learning (RL).

There are going to be two differences from the way we’ve done our past assignments.

*   First, we are not going to be using our language model codebase and models from earlier. We would ideally like to use base language models trained from previous assignments, but finetuning those models will not give us a satisfying result—these models are far too weak to display non-trivial mathematical reasoning capabilities. Because of this, we are going to switch to a modern, high-performance language model that we can access (Qwen 2.5 Math 1.5B Base) and do most of our work on top of that model.
*   Second, we are going to introduce a new benchmark with which to evaluate our language models. Up until this point, we have embraced the view that cross-entropy is a good surrogate for many downstream tasks. However, the point of this assignment will be to bridge the gap between base models and downstream tasks and so we will have to use evaluations that are separate from cross-entropy. We will use the MATH 12K dataset from Hendrycks et al. [2021], which consists of challenging high-school competition mathematics problems. We will evaluate language model outputs by comparing them against a reference answer.

## 2.2 Chain-of-Thought Reasoning and Reasoning RL

An exciting recent trend in language models is the use of chain-of-thought reasoning to improve performance across a variety of tasks. Chain-of-thought refers to the process of reasoning through a problem step-by-step, generating intermediate reasoning steps before arriving at a final answer.

**Chain-of-thought reasoning with LLMs.** Early chain-of-thought approaches finetuned language models to solve simple mathematical tasks like arithmetic by using a “scratchpad” to break the problem into intermediate steps [Nye et al., 2021]. Other work prompts a strong model to “think step by step” before answering, finding that this significantly improves performance on mathematical reasoning tasks like grade-school math questions [Wei et al., 2023].

**Learning to reason with expert iteration.** The Self-Taught Reasoner (STaR) [Zelikman et al., 2022] frames reasoning as a bootstrapping loop: a pretrained model first samples diverse chains-of-thought (CoTs), keeps only those that lead to correct answers, and then finetunes on these “expert” traces. Iterating this cycle can improve the LM’s reasoning capabilities and solve rate. STaR demonstrated that this version of expert iteration [Anthony et al., 2017] using automatic, string match–based verification of generated answers can bootstrap reasoning skills without human-written reasoning traces.

**Reasoning RL with verified rewards, o1, and R1.** Recent work has explored using more powerful reinforcement learning algorithms with verified rewards to improve reasoning performance. OpenAI’s o1 (and subsequent o3/o4) [OpenAI et al., 2024], DeepSeek’s R1 [DeepSeek-AI et al., 2025], and Moonshot’s kimi k1.5 [Team et al., 2025] use policy gradient methods [Sutton et al., 1999] to train on math and code tasks where string matching or unit tests verify correctness, demonstrating remarkable improvements in competition math and coding performance. Later works such as Open-R1 [Face, 2025], SimpleRL-Zoo [Zeng et al., 2025], and TinyZero [Pan et al., 2025] confirm that pure reinforcement learning with verified rewards—even on models as small as 1.5B parameters—can improve reasoning performance.

**Our setup: model and dataset.** In the following sections, we will consider progressively more complex approaches to train a base language model to reason step-by-step in order to solve math problems. For this assignment, we will be using the Qwen 2.5 Math 1.5B Base model, which was continually pretrained from the Qwen 2.5 1.5B model on high-quality synthetic math pretraining data [Yang et al., 2024]. The MATH dataset is available on the Together cluster at `/data/a5-alignment/MATH`.

> **Tip for Open-Source Auditors: Alternative Datasets**
>
> Unfortunately, the MATH dataset is not publicly available due to a copyright claim. If you are following along at home, you can use one of the following open-source mathematical reasoning datasets:
>
> *   Countdown [Pan et al., 2025], available here: a simple synthetic task based on the British TV show Countdown that has served as a popular testbed for small-scale reasoning RL.
> *   GSM8K [Cobbe et al., 2021a], available here: grade-school math problems, which are easier than MATH but should allow you to debug correctness and get familiar with the reasoning RL pipeline.
> *   Tulu 3 SFT Math [Lambert et al., 2025], available here: synthetic math problems generated using GPT-4o and Claude 3.5 Sonnet. Because these are synthetic, some answers (or even the questions) may not be entirely correct.
> *   Some other math SFT dataset linked here.
>
> To obtain short ground-truth labels (e.g., 1/2) if they are not provided directly, you can process the ground-truth column with a math answer parser such as Math-Verify.
