# Supervised Fine-Tuning (SFT) Tutorial

This tutorial walks you through supervised fine-tuning (SFT) of a large language model using Megatron-LM, with **Qwen/Qwen3-0.6B** as the example model. SFT adapts a pretrained model to follow instructions by training on curated prompt–response pairs.

## Overview

The SFT workflow in Megatron-LM consists of the following steps:

1. **Environment Setup** – Install dependencies and prepare the runtime.
2. **Download the Pretrained Model** – Obtain Qwen3-0.6B weights from Hugging Face.
3. **Prepare the SFT Dataset** – Format your data as JSONL with chat-style messages.
4. **Launch Fine-Tuning** – Run the SFT training script with the appropriate configuration.
5. **Evaluate the Model** – Generate text to verify training results.

## Prerequisites

- **Hardware**: At least 1 NVIDIA GPU (Ampere or newer recommended for BF16 support).
- **Software**: Python 3.10+, PyTorch 2.0+, CUDA 12+.
- **Megatron-LM**: A working installation of this repository (see [Installation Guide](../get-started/install.md)).

Install the required Python packages:

```bash
pip install nvidia-modelopt datasets transformers sentencepiece
```

## Step 1: Download the Pretrained Model

Download the Qwen3-0.6B model checkpoint from Hugging Face:

```bash
# Install huggingface_hub CLI if not already installed
pip install huggingface_hub

# Download the model (requires ~1.5 GB disk space)
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /path/to/Qwen3-0.6B
```

Set the checkpoint path as an environment variable:

```bash
export HF_MODEL_CKPT=/path/to/Qwen3-0.6B
```

## Step 2: Prepare the SFT Dataset

Megatron-LM SFT supports datasets in **JSONL** format where each line contains a `messages` field with a list of chat-style messages. Each message has a `role` (`system`, `user`, or `assistant`) and `content`.

### Dataset Format

Create a file named `sft_data.jsonl`. Each line should follow this structure:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain gravity in simple terms."}, {"role": "assistant", "content": "Gravity is the force that pulls objects toward each other. It's what keeps us on the ground and what makes things fall when you drop them."}]}
```

**Key points:**

- Each conversation starts with an optional `system` message, followed by alternating `user` and `assistant` turns.
- Multiple turns are supported within a single conversation.
- Only the `assistant` portions are used for computing the training loss (the `system` and `user` portions are masked).

### Using a Hugging Face Dataset

Alternatively, you can use an existing Hugging Face dataset directly. The ModelOpt SFT scripts support loading datasets from the Hugging Face Hub by name. For example, the default dataset used by the fine-tuning script is `Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered`.

Set a custom dataset path:

```bash
export DATASET="Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered"
```

Or point to a local JSONL file:

```bash
export DATASET="/path/to/sft_data.jsonl"
```

## Step 3: Launch SFT Training

### Option A: Using the ModelOpt SFT Script (Recommended)

The simplest way to run SFT is with the integrated ModelOpt fine-tuning script, which handles checkpoint conversion from Hugging Face format automatically.

Navigate to the examples directory and run:

```bash
cd examples/post_training/modelopt

TP=1 \
HF_MODEL_CKPT=/path/to/Qwen3-0.6B \
MLM_MODEL_SAVE=/path/to/output/Qwen3-0.6B-sft \
DATASET="Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered" \
./finetune.sh Qwen/Qwen3-0.6B
```

**Key environment variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_MODEL_CKPT` | Path to the downloaded Hugging Face checkpoint | Model name from config |
| `MLM_MODEL_SAVE` | Directory to save the fine-tuned Megatron checkpoint | Same as `MLM_MODEL_CKPT` |
| `DATASET` | Hugging Face dataset name or local JSONL path | `Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered` |
| `TP` | Tensor parallelism degree | `1` |
| `PP` | Pipeline parallelism degree | `1` |

### Option B: Using torchrun Directly

For more control, you can call the fine-tuning script directly with `torchrun`:

```bash
cd examples/post_training/modelopt

torchrun --nproc_per_node=1 finetune.py \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 28 \
    --hidden-size 1024 \
    --ffn-hidden-size 3072 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --qk-layernorm \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/Qwen3-0.6B \
    --make-vocab-size-divisible-by 1187 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --no-bias-swiglu-fusion \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --train-samples 128000 \
    --lr-decay-samples 128000 \
    --lr-warmup-samples 0 \
    --split 100,0,0 \
    --finetune-hf-dataset "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered" \
    --lr 5.0e-5 \
    --min-lr 1.0e-7 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --reset-position-ids \
    --reset-attention-mask \
    --eod-mask-loss \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-check-for-nan-in-loss-and-grad \
    --eval-iters 1 \
    --eval-interval 1000 \
    --save-interval 1000 \
    --log-interval 100 \
    --load /path/to/Qwen3-0.6B \
    --save /path/to/output/Qwen3-0.6B-sft \
    --finetune \
    --auto-detect-ckpt-format \
    --export-te-mcore-model
```

> **Note:** The model architecture arguments (e.g., `--num-layers`, `--hidden-size`) must match the Qwen3-0.6B configuration exactly. These values are sourced from `examples/post_training/modelopt/conf/Qwen/Qwen3-0.6B.sh`. In particular, `--max-position-embeddings 40960` is the Qwen3-0.6B model's native RoPE context window (larger than `--seq-length` to allow for extended context), and `--make-vocab-size-divisible-by 1187` ensures the padded vocabulary size matches the model's embedding table.

## Step 4: Customizing Training Hyperparameters

You can customize training by setting environment variables before launching the script:

```bash
cd examples/post_training/modelopt

# Custom training configuration
MLM_DATA_ARGS=" \
    --train-samples 50000 \
    --lr-decay-samples 50000 \
    --lr-warmup-samples 500 \
    --split 100,0,0 \
    --finetune-hf-dataset /path/to/sft_data.jsonl \
" \
MLM_OPTIM_ARGS=" \
    --lr 2.0e-5 \
    --min-lr 1.0e-7 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
" \
TP=1 \
HF_MODEL_CKPT=/path/to/Qwen3-0.6B \
MLM_MODEL_SAVE=/path/to/output/Qwen3-0.6B-sft \
./finetune.sh Qwen/Qwen3-0.6B
```

### Hyperparameter Recommendations

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| Learning rate (`--lr`) | `1e-5` to `5e-5` | Lower values for smaller datasets |
| Weight decay (`--weight-decay`) | `0.0` to `0.01` | Light regularization |
| Warmup samples (`--lr-warmup-samples`) | 0–500 | Short warmup is typical for SFT |
| Sequence length (`--seq-length`) | `2048` to `4096` | Balance between context length and memory |
| LR schedule (`--lr-decay-style`) | `cosine` | Cosine decay is standard |
| Gradient clipping (`--clip-grad`) | `1.0` | Prevents gradient explosion |

## Step 5: Multi-GPU Training

For training on multiple GPUs, set the tensor parallelism (`TP`) or use data parallelism. Qwen3-0.6B is small enough to fit on a single GPU, so multi-GPU setups primarily benefit from data parallelism for faster training.

### Data Parallel on 4 GPUs

```bash
cd examples/post_training/modelopt

TP=1 \
DP=4 \
HF_MODEL_CKPT=/path/to/Qwen3-0.6B \
MLM_MODEL_SAVE=/path/to/output/Qwen3-0.6B-sft \
./finetune.sh Qwen/Qwen3-0.6B
```

### Tensor Parallel on 2 GPUs

```bash
cd examples/post_training/modelopt

TP=2 \
HF_MODEL_CKPT=/path/to/Qwen3-0.6B \
MLM_MODEL_SAVE=/path/to/output/Qwen3-0.6B-sft \
./finetune.sh Qwen/Qwen3-0.6B
```

## Step 6: Generate Text from the Fine-Tuned Model

After training completes, verify the results by generating text:

```bash
cd examples/post_training/modelopt

TP=1 \
MLM_MODEL_CKPT=/path/to/output/Qwen3-0.6B-sft \
./generate.sh Qwen/Qwen3-0.6B
```

## Step 7: Evaluate the Fine-Tuned Model

Run MMLU evaluation to measure model quality:

```bash
cd examples/post_training/modelopt

TP=1 \
MLM_MODEL_CKPT=/path/to/output/Qwen3-0.6B-sft \
./mmlu.sh Qwen/Qwen3-0.6B
```

## Using the Core SFT Dataset Class

For advanced use cases, Megatron-LM also provides a built-in `SFTDataset` class at `megatron/training/datasets/sft_dataset.py`. This class:

- Loads JSONL data with the `messages` field.
- Supports multi-turn conversations with `system`, `user`, and `assistant` roles.
- Automatically packs multiple conversations into fixed-length sequences for efficient training.
- Masks prompt tokens so that loss is computed only on assistant responses.
- Supports context parallelism with appropriate padding.

To use it in a custom training script, the dataset expects JSONL files where each line has a `messages` key:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory (OOM) | Reduce `--seq-length`, or increase `TP` to shard model across more GPUs |
| Slow training | Increase `DP` to use more GPUs for data parallelism |
| NaN in loss | Reduce `--lr`, or increase `--clip-grad` |
| Tokenizer errors | Ensure `--tokenizer-model` points to the correct Hugging Face model directory |
| Checkpoint format errors | Use `--auto-detect-ckpt-format` to handle format detection automatically |

## Next Steps

- **Quantization**: Apply post-training quantization for efficient deployment. See the [ModelOpt README](../../../examples/post_training/modelopt/README.md).
- **Parallelism**: Learn about scaling to larger models with [Parallelism Strategies](parallelism-guide.md).
- **Data Preparation**: For pretraining data, see the [Data Preparation Guide](data-preparation.md).
- **Reinforcement Learning**: For RLHF-based fine-tuning, see [Megatron RL](features/megatron_rl.md).
