# SFT 脚本代码拆解 (Supervised Finetuning GPT)

本文档按模块拆解 `examples/post_training/modelopt/finetune.py` 的整段 SFT 流程，说明每个部分的职责和数据流。

---

## 1. 入口与依赖

```python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
```

- 把仓库根目录加入 `sys.path`，以便 `from megatron...` 和 `from model_provider import model_provider` 能正确解析。
- **Megatron Core**：`mpu`（并行拓扑）、`tensor_parallel`（TP 通信）、`GPTModel`、`ModelType`。
- **Post-training**：`add_modelopt_args`、`loss_func`、`modelopt_gpt_mamba_builder`、`report_draft_acceptance_length`、`model_provider`。
- **Training 框架**：`get_args`、`get_timers`、`get_tokenizer`、`pretrain`，以及 `get_batch_on_this_cp_rank`、`get_ltor_masks_and_position_ids`、`print_rank_0`。

整体：在 Megatron 的 `pretrain()` 框架下，用「SFT 数据集 + 自定义 get_batch/forward_step」做监督微调；模型由 `model_provider(modelopt_gpt_mamba_builder)` 构建，支持 GPT / Mamba 与 ModelOpt 特性。

---

## 2. 常量与工具函数

### `REMOVE_THINK_CHAT_TEMPLATE`

- 一段 Jinja2 模板片段：若 `content` 里包含 `</think>
`，则只保留该标签**之后**的文本。
- 用途：DeepSeek-V3/R1 等带“思考”的模型，在 SFT 时只对“去掉思考后的回答”算 loss；在 `SFTDataset.__init__` 里会从 tokenizer 的 `chat_template` 中**删掉**这段，从而在 tokenize 时保留 `<think>...</think>
` 段（即对“思考”部分也参与训练）。

### `add_finetune_args(parser)`

- 增加 SFT/离线蒸馏相关参数：
  - `--offline-distillation-data`：离线蒸馏特征所在目录（train/valid/test 子目录）。
- 并调用 `add_modelopt_args(parser)`，因此会一并注册：
  - `--finetune-hf-dataset`、`--finetune-data-split`
  - `--export-offline-model`（离线 speculative 学生模型，无 decoder 层）
  - 以及 ModelOpt 的 export/quantization/distillation 等参数。

### `get_eos_id()`

- 返回「用于样本之间分隔」的 EOS token id。
- 原因：在 packing 时会在两条样本之间插入 EOS；若 tokenizer 的 `eos_token` 本身会出现在对话内容里，就需要换成一个不会出现在消息里的特殊 id，避免把“内容里的 EOS”当成“样本结束”。
- 逻辑：根据 `tokenizer._tokenizer.eos_token` 的字符串匹配（如 `<|eot_id|>`、`<|im_end|>` 等）返回对应 id，否则返回 `eos_token_id`。不同模型用不同占位 id（128001、200001、151643、199999 等）。

---

## 3. 数据集

### `OfflineDataset`

- **用途**：离线蒸馏 / 离线 speculative 训练。数据已由 base 模型预先跑出特征并落盘。
- **构造**：`data_dir` 下每个文件视为一个样本；`num_samples` 为“逻辑长度”（可大于文件数，用于控制训练步数）。
- **`__getitem__(idx)`**：`idx % len(file_paths)` 选文件，`torch.load(file_path)` 得到一个样本（通常含 `input_ids`、`aux_hidden_states`、`hidden_states` 等）。
- 与在线 SFT 的区别：不在这里做 tokenize 和 packing，只读预计算好的张量。

### `SFTDataset`

在线 SFT 的主数据集：从 HuggingFace 或本地读原始对话，按 `seq_length` 做 packing，并生成 `loss_mask`（只在 assistant 回复上算 loss）。

#### 类属性（数据集名 → 行为）

- **`hf_dataset_to_kwargs`**：不同 HF 数据集加载时的 `load_dataset(..., **kwargs)`，例如 `split="train"` 或 `split="train_sft"`。
- **`hf_dataset_to_conversation`**：把原始格式转成“OpenAI 风格”的 `conversations` 列表（每项 `role` + `content`）。例如：
  - OpenOrca：`question`/`response` → `[user, assistant]`；
  - ShareGPT 系（Daring-Anteater、Magpie 等）：`from`/`value` 映射到 `user`/`assistant`/`system`。
- **`hf_dataset_to_prompt_template`**：仅个别数据集用自定义 prompt 模板（如 OpenOrca 的 `{{ messages['question'] + ' ' + ... }}`），多数用 tokenizer 自带的 `chat_template`。

#### `__init__(num_packed_samples, hf_dataset, tokenizer, seq_length, num_shards, shard_index)`

- 校验 `tokenizer` 为 `transformers.PreTrainedTokenizerBase`。
- 从 `chat_template` 里去掉 `REMOVE_THINK_CHAT_TEMPLATE`（见上），以便保留 `<think>` 段。
- `datasets.load_dataset(hf_dataset, ...)` 按 shard 切分：每个 DP rank 只取一份 shard（`num_shards` / `shard_index`），避免重复数据。
- 根据 `hf_dataset` 选择 `data_transformation`（上面提到的 conversation 转换）；若 tokenizer 没有 `chat_template`，则用 `hf_dataset_to_prompt_template`。
- **不在这里**做 packing：packed 结果缓存在 `self.indexed_dataset`，在首次 `__getitem__` 访问时按需填充。

#### `__len__` / `__getitem__(idx)`

- `__len__`：返回 `num_packed_samples`（训练步数 × micro_batch_size 等推导出的“打包样本数”）。
- `__getitem__(idx)`：
  - 先按 shard 映射：`idx = idx // num_shards`。
  - 若 `idx >= len(indexed_dataset)`，循环调用 `_process_and_pack_example()` 直到够用（或原始数据用完返回 `None` 则停）。
  - 用 `idx % len(indexed_dataset)` 取一条打包样本，把 list 转成 `LongTensor` 返回；每条包含 `input_ids`、`loss_mask`（以及 packed 时的 `token_count` 等）。

#### `_process_and_pack_example()`

- 目标：凑满 `seq_length + 1` 个 token（多出的 1 是给 labels 用的右移）。
- 循环取 `_raw_samples[_raw_sample_index]`，经 `_process_example()` 得到 `input_ids`、`loss_mask`、`token_count`；若有效就加入当前 pack，累加 token 数直到 ≥ `seq_length + 1`。
- 把多个样本的 `input_ids`、`loss_mask` 分别 concat 成一条；返回一个 packed 字典（list 形式，后面在 `__getitem__` 里再转 Tensor）。

#### `_process_example(example)`

- 先用 `data_transformation(example)` 转成 OpenAI 风格（若需要）。
- 取 `conversations` 或 `messages`；若少于 2 条或首条是 assistant，则丢弃（return None）。
- `tokenizer.apply_chat_template(...)` 得到 `input_ids`；构造 `loss_mask`：与 `input_ids` 同长，**仅 assistant 回复位置为 1**，其余（含 system/user、EOS 分隔）为 0。具体实现里会在末尾加一个 EOS 且对应 loss_mask 为 0。
- 若长度超过 `seq_length` 则截断；返回 `{ 'input_ids', 'loss_mask', 'token_count' }`。

#### 静态方法

- `_to_conversation(question, response)`：单轮 QA → `[user, assistant]`。
- `_sharegpt_to_openai_conversations(data)`：ShareGPT 的 `from`/`value` → `role`/`content`，并做大小写归一化（Human→user、Assistant→assistant 等）。
- `_special_to_openai_conversations(data)`：从 `data["input"]["messages"]` 取出直接当 `conversations`。

---

## 4. 数据集提供方：`train_valid_test_sft_datasets_provider`

- 入参：`train_val_test_num_samples`，即 `[train_samples, valid_samples, test_samples]`。
- 若 **`args.export_offline_model`**：
  - 使用 `OfflineDataset`，数据目录为 `args.offline_distillation_data` 下的 `train` / `valid` / `test` 子目录。
- 否则：
  - 使用 `SFTDataset`，参数来自 `args`：`finetune_hf_dataset`、`tokenizer._tokenizer`、`seq_length`，以及 `mpu.get_expert_data_parallel_world_size/rank()` 做 shard。
- 约束：`micro_batch_size` 必须为 1（SFT dataloader 当前实现只支持 1）。

---

## 5. Batch 构造：`get_batch(data_iterator)`

- 中间 pipeline stage 不参与数据消费，直接返回 `(None,)*5`。
- 从 `data_iterator` 取一个 batch（字典）；若为**在线 SFT**，用 `tensor_parallel.broadcast_data` 在 TP 组内广播 `input_ids`、`loss_mask`；若为**离线**，则广播 `input_ids` 并补全 `loss_mask`，再广播 `aux_hidden_states`、`hidden_states`（base 模型算好的隐状态）。
- 从 `data_b` 解出：
  - `tokens` = `input_ids[:, :seq_length]`
  - `labels` = `input_ids[:, 1:seq_length+1]`（右移 1）
  - `answer_only_loss_mask` = `loss_mask[:, 1:seq_length+1]`
- 调用 `get_ltor_masks_and_position_ids(tokens, eos_id, ...)` 得到 `attention_mask`、`loss_mask`、`position_ids`；再把 `loss_mask` 与 `answer_only_loss_mask` 相乘，得到“仅回答部分”的 loss mask。
- 组装 `batch` 字典：`tokens`、`labels`、`loss_mask`、`attention_mask`、`position_ids`；离线时额外加入 `aux_hidden_states`、`hidden_states`（并做 transpose/slice 以对齐 sequence 维）。
- 最后用 `get_batch_on_this_cp_rank(batch)` 按 context parallel rank 切分 sequence，返回给当前 rank 使用的 batch。

---

## 6. 非 loss 回调：`non_loss_data_func(model)`

- 仅在**非离线**且 **context_parallel_size == 1** 时调用 `report_draft_acceptance_length(model)`，用于 speculative decoding 相关统计。
- 若调用异常只打印，不中断训练。

---

## 7. 前向与 loss：`forward_step(data_iterator, model)`

- 用 `get_batch(data_iterator)` 拿到当前 batch。
- **在线 SFT**：`output_tensor = model(tokens, position_ids, attention_mask, labels=labels)`。
- **离线**：多传 `aux_hidden_states`、`hidden_states`，即 `model(..., aux_hidden_states=..., hidden_states=...)`（学生模型用 base 的隐状态做蒸馏/辅助）。
- 返回 `(output_tensor, partial(loss_func, loss_mask, model=model))`。训练框架会再调用这个 partial，把 `output_tensor` 和 `loss_mask` 传给 `loss_func` 得到标量 loss（内部会做 mask、以及可选的 KD 等）。

---

## 8. 主入口：`pretrain(...)`

```python
pretrain(
    train_valid_test_sft_datasets_provider,
    partial(model_provider, modelopt_gpt_mamba_builder),
    ModelType.encoder_or_decoder,
    forward_step,
    extra_args_provider=add_finetune_args,
    args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    non_loss_data_func=non_loss_data_func,
)
```

- **数据集**：由 `train_valid_test_sft_datasets_provider` 根据 `export_offline_model` 和 `offline_distillation_data` / `finetune_hf_dataset` 决定用 `OfflineDataset` 还是 `SFTDataset`。
- **模型**：`model_provider` 绑定 `modelopt_gpt_mamba_builder`，即用 ModelOpt 的 GPT/Mamba builder（支持 HF 转 Megatron、量化、蒸馏、离线学生等）。
- **类型**：`encoder_or_decoder`（因果 LM）。
- **前向与 loss**：`forward_step` + `loss_func`（含可选 KD）。
- **参数**：通过 `add_finetune_args` 注册 SFT/离线蒸馏和 ModelOpt 相关参数；默认使用 `HuggingFaceTokenizer`。

---

## 9. 数据流小结

1. **在线 SFT**：HF 数据集名/路径 → `SFTDataset` 按 DP shard 加载 → 每条样本经 conversation 转换 + `apply_chat_template` → packing 成 `seq_length` → `get_batch` 里广播、切 CP、生成 masks/labels → `forward_step` 只算 LM loss（且仅 assistant 部分由 `loss_mask` 参与）。
2. **离线蒸馏/Speculative**：预计算好的特征目录 → `OfflineDataset` 按文件 load → `get_batch` 广播 tokens + 隐状态 → `forward_step` 把 base 的 `aux_hidden_states`/`hidden_states` 喂给学生模型，loss 可由 `loss_func` 里的 KD 等扩展。

整体上，这份脚本把「数据格式与 packing」和「在线/离线两种数据源」封装在 Dataset 层，把「batch 形状与并行」放在 `get_batch`，把「模型前向与 loss 形式」放在 `forward_step` + `loss_func`，其余由 Megatron 的 `pretrain()` 统一调度（optimizer、DDP、checkpoint、logging 等）。
