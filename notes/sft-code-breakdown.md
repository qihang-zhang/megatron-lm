# SFT 脚本代码详解（带行号与代码块）

本文档对 `examples/post_training/modelopt/finetune.py` 做逐段解读，每段附**行号范围**和**对应代码块**，便于对照源码阅读。

---

## 1. 文件头、导入与路径 (L1–31)

### 1.1 版权与 docstring (L1–3)

```1:3:examples/post_training/modelopt/finetune.py
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Supervised Finetuning GPT."""
```

- 说明本文件是「监督微调 GPT」的入口脚本。

### 1.2 标准库与类型 (L4–9)

```4:9:examples/post_training/modelopt/finetune.py
import itertools
import json
import os
import sys
from functools import partial
from typing import Any, Dict, Optional
```

- `itertools`：在 packing 时用 `chain.from_iterable` 拼接多条样本的 `input_ids` / `loss_mask`。
- `json`：未在后续使用，可能是历史遗留。
- `os`：环境变量（如 `HF_TOKEN`）、路径拼接、列目录（OfflineDataset）。
- `sys`：用于 `sys.path.append`。
- `partial`：在 `pretrain()` 里绑定 `model_provider(modelopt_gpt_mamba_builder)`，在 `forward_step` 返回值里绑定 `loss_func` 的 `loss_mask` 和 `model`。
- `Any, Dict, Optional`：类型注解（如 `_process_example(example: Dict[str, Any])`）。

### 1.3 仓库根路径 (L11)

```11:11:examples/post_training/modelopt/finetune.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
```

- `__file__` 为 `.../examples/post_training/modelopt/finetune.py`，上溯三层得到仓库根目录并加入 `sys.path`，使 `from megatron...` 和 `from model_provider import model_provider` 能正确解析（脚本通常从 `examples/post_training/modelopt/` 或仓库根目录运行）。

### 1.4 第三方与 Megatron 导入 (L13–29)

```13:29:examples/post_training/modelopt/finetune.py
import datasets
import torch
import transformers

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.loss_func import loss_func
from megatron.post_training.model_builder import modelopt_gpt_mamba_builder
from megatron.post_training.non_loss_data_func import report_draft_acceptance_length
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    print_rank_0,
)
from model_provider import model_provider
```

- **datasets**：`SFTDataset` 里用 `datasets.load_dataset()` 加载 HuggingFace 数据集。
- **torch**：张量、`Dataset`、`torch.load` 等。
- **transformers**：`PreTrainedTokenizerBase` 类型检查与 tokenizer 的 `apply_chat_template`。
- **mpu**：获取 expert data parallel 的 world size / rank，用于数据集 shard。
- **tensor_parallel**：TP 组内 `broadcast_data`，把 data iterator 只在 rank0 上的数据广播到所有 TP rank。
- **ModelType**：传给 `pretrain()`，此处为 `encoder_or_decoder`（因果 LM）。
- **GPTModel**：类型注解（如 `forward_step(..., model: GPTModel)`）。
- **add_modelopt_args**：注册 `--finetune-hf-dataset`、`--export-offline-model` 等 ModelOpt/SFT 参数。
- **loss_func**：对 `output_tensor` 与 `loss_mask` 做 mask 后的 loss 计算（含可选 KD）。
- **modelopt_gpt_mamba_builder**：构建支持 ModelOpt 的 GPT/Mamba 模型（HF 转换、量化、蒸馏、离线学生等）。
- **report_draft_acceptance_length**：speculative decoding 相关统计，在 `non_loss_data_func` 里调用。
- **get_args, get_timers, get_tokenizer, pretrain**：Megatron 训练框架的全局参数、计时、tokenizer、主训练循环。
- **get_batch_on_this_cp_rank**：按 context parallel rank 切分 batch 的 sequence 维。
- **get_ltor_masks_and_position_ids**：根据 `tokens` 和 EOS 生成 `attention_mask`、`loss_mask`、`position_ids`（left-to-right causal）。
- **print_rank_0**：仅 rank0 打印，用于数据集构建完成等日志。
- **model_provider**：当前目录下的 `model_provider`，与 `modelopt_gpt_mamba_builder` 组合后交给 `pretrain()`。

---

## 2. 常量与 EOS 占位 (L31–62)

### 2.1 REMOVE_THINK_CHAT_TEMPLATE (L31–33)

```31:33:examples/post_training/modelopt/finetune.py
REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)
```

- 一段 **Jinja2** 片段：若 `content` 中包含 `</think>
`，则只保留该标签**之后**的文本。
- 用途：DeepSeek-V3/R1 等带“思考”的模型，在 tokenizer 的 `chat_template` 里常带这段逻辑，用于推理时只输出“回答部分”。SFT 时若希望**保留** `<think>...</think>
` 参与训练，就在 `SFTDataset.__init__` 里把 `chat_template` 中的这段**整体替换成空字符串**（见 L163–164），这样 tokenize 时不会截掉思考段。

### 2.2 add_finetune_args (L36–44)

```36:44:examples/post_training/modelopt/finetune.py
def add_finetune_args(parser):
    """Add additional arguments for finetune."""
    group = parser.add_argument_group(title='Finetune')
    group.add_argument("--offline-distillation-data", type=str, help="Path to the offline dataset directory with base model features.")


    add_modelopt_args(parser)
    return parser
```

- 新增参数组 `Finetune`，只增加一个参数：`--offline-distillation-data`，指向**离线蒸馏**特征根目录（其下应有 `train` / `valid` / `test` 子目录，见 L354–356）。
- `add_modelopt_args(parser)` 会继续注册：`--finetune-hf-dataset`、`--finetune-data-split`、`--export-offline-model`，以及 ModelOpt 的 export/quantization/distillation 等。因此 SFT 与离线蒸馏所需的 CLI 参数都从这里进入。

### 2.3 get_eos_id (L45–62)

```45:62:examples/post_training/modelopt/finetune.py
def get_eos_id():
    """Return the eos token id.

    We insert eos_token between two samples during packing. However, if the eos_token is used in message or after turns,
    we need to replace it with some other special tokens that do not appear in message."""
    tokenizer = get_tokenizer()
    hf_tokenizer = tokenizer._tokenizer

    if hf_tokenizer.eos_token == "<|eot_id|>":
        return 128001
    if hf_tokenizer.eos_token == "<|eot|>":
        return 200001
    if hf_tokenizer.eos_token == "<|im_end|>":
        return 151643
    if hf_tokenizer.eos_token == "<|return|>":
        return 199999

    return hf_tokenizer.eos_token_id
```

- 返回用于「样本之间分隔」的 **EOS token id**。在 packing 时会在两条样本之间插入一个 EOS；若 tokenizer 的 `eos_token` 也会出现在对话内容里，就需要换成一个**不会出现在消息里**的 id，避免把内容里的 EOS 当成样本边界。
- 逻辑：根据 `hf_tokenizer.eos_token` 的**字符串**匹配已知模型（如 Qwen、DeepSeek、Llama 等），返回对应占位 id；否则退回 `hf_tokenizer.eos_token_id`。不同模型占位 id 不同（128001、200001、151643、199999 等），以避免与真实词表冲突。

---

## 3. OfflineDataset (L65–84)

```65:84:examples/post_training/modelopt/finetune.py
class OfflineDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, num_samples):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.file_paths = []

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isfile(item_path):
                self.file_paths.append(item_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = idx % len(self.file_paths)
        file_path = self.file_paths[idx]
        sample = torch.load(file_path)
        return sample
```

- **用途**：离线蒸馏 / 离线 speculative 训练。数据已由 base 模型预先跑出并写入单个文件（每个文件一个样本），此处只做「按索引读文件」。
- **__init__**：`data_dir` 下所有**文件**的路径存入 `file_paths`（不递归子目录）；`num_samples` 为逻辑长度，可大于文件数，用于控制训练步数（dataloader 会按 `num_samples` 次访问）。
- **__len__**：返回 `num_samples`。
- **__getitem__(idx)**：`idx % len(self.file_paths)` 选文件，`torch.load(file_path)` 得到一条样本；通常包含 `input_ids`、`aux_hidden_states`、`hidden_states` 等，供 `get_batch` 和 `forward_step` 使用。不做 tokenize 与 packing。

---

## 4. SFTDataset：类属性与工具方法 (L85–331)

### 4.1 数据集名 → load_dataset kwargs (L87–93)

```87:93:examples/post_training/modelopt/finetune.py
    hf_dataset_to_kwargs = {
        "Open-Orca/OpenOrca": {"split": "train"},
        "Open-Orca/SlimOrca": {"split": "train"},
        "nvidia/Daring-Anteater": {"split": "train"},
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": {"split": "train"},
        "HuggingFaceH4/ultrachat_200k": {"split": "train_sft"},
    }
```

- 不同 HuggingFace 数据集在调用 `datasets.load_dataset(hf_dataset, **kwargs)` 时使用的 `kwargs`；多数用 `split="train"`，ultrachat 用 `train_sft`。未出现在此字典中的数据集默认使用 `{"split": "train"}`（见 L164–165）。

### 4.2 数据集名 → 对话格式转换 (L95–104)

```95:104:examples/post_training/modelopt/finetune.py
    hf_dataset_to_conversation = {
        "Open-Orca/OpenOrca": lambda data: SFTDataset._to_conversation(
            data["question"], data["response"]
        ),
        "Open-Orca/SlimOrca": lambda data: SFTDataset._sharegpt_to_openai_conversations(data),
        "nvidia/Daring-Anteater": lambda data: SFTDataset._sharegpt_to_openai_conversations(data),
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": lambda data: SFTDataset._sharegpt_to_openai_conversations(
            data
        ),
    }
```

- 把**原始样本**转成「OpenAI 风格」的对话：即 `conversations` 或 `messages` 列表，每项为 `{"role": "user"|"assistant"|"system", "content": "..."}`。
- OpenOrca：`question` / `response` → 单轮 `[user, assistant]`（`_to_conversation`）。
- SlimOrca、Daring-Anteater、Magpie 等：ShareGPT 格式（`from` / `value`）→ `_sharegpt_to_openai_conversations` 做 role 归一化并输出 `conversations`。

### 4.3 数据集名 → 自定义 prompt 模板 (L106–108)

```106:108:examples/post_training/modelopt/finetune.py
    hf_dataset_to_prompt_template = {
        "Open-Orca/OpenOrca": "{{ messages['question'] + ' ' + messages['response'] + ' ' }}",
    }
```

- 仅当 tokenizer 的 `chat_template` 为 `None` 时，会按数据集名从这里取模板（见 L183–184）；目前只有 OpenOrca 使用自定义 Jinja2 模板，直接拼 `question` 和 `response`。其他数据集依赖 tokenizer 自带的 `chat_template`。

### 4.4 _wildcard_get (L110–117)

```110:117:examples/post_training/modelopt/finetune.py
    @classmethod
    def _wildcard_get(cls, directory: Dict[str, Any], name: str, default_value=None):
        ret = default_value
        for key, val in directory.items():
            if key in name:
                ret = val
                break
        return ret
```

- 在字典 `directory` 中查找「key 是 name 的子串」的项，返回对应 value；若没有则返回 `default_value`。用于通过「数据集名或路径」匹配 `hf_dataset_to_conversation`（例如本地路径包含 `Magpie-Llama` 时仍能命中 `Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered` 的转换逻辑）。

### 4.5 __init__ (L119–193)

```119:165:examples/post_training/modelopt/finetune.py
    def __init__(
        self,
        num_packed_samples: int,
        hf_dataset: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
        seq_length: int,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        """A simple dataset implementation for supervised fine-tuning.
        ...
        """
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

        self.num_packed_samples = num_packed_samples
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data_transformation = lambda data: data
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.indexed_dataset = []
        self._raw_sample_index = 0

        # [WAR]: For DeepSeek-V3/R1 tokenizer, we modify the chat_template such that the <think>
        # tokens are preserved for supervised learning.
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            REMOVE_THINK_CHAT_TEMPLATE, ""
        )

        hf_dataset_kwargs = SFTDataset.hf_dataset_to_kwargs.get(
            self.hf_dataset, {"split": "train"}
        )
        self._raw_samples = datasets.load_dataset(self.hf_dataset, token=os.environ.get("HF_TOKEN", None), **hf_dataset_kwargs)
        self._raw_samples = self._raw_samples.shard(
            num_shards=self.num_shards, index=shard_index
        )
```

- **参数**：`num_packed_samples` 为「打包后的样本数」（训练步数相关）；`hf_dataset` 为 HF 名或本地路径；`tokenizer` 必须是 `PreTrainedTokenizerBase`；`seq_length` 为最大序列长度；`num_shards` / `shard_index` 用于数据并行下的数据切分。
- **data_transformation** 默认恒等；若下面匹配到 `hf_dataset_to_conversation` 会覆盖为对应 lambda。
- **indexed_dataset** 存已打包的样本（list of dict），**按需**在 `__getitem__` 里填充；**\_raw_sample_index** 指向下一条要处理的原始样本。
- **chat_template**：从 tokenizer 的 `chat_template` 里去掉 `REMOVE_THINK_CHAT_TEMPLATE`，使 DeepSeek 等「思考」段在 SFT 时被保留。
- **load_dataset**：用 `hf_dataset_to_kwargs.get(..., {"split": "train"})` 的 kwargs 加载；再用 `shard(num_shards, index)` 让每个 data parallel rank 只取一份，避免重复。

```171:193:examples/post_training/modelopt/finetune.py
        print(
            "Rank {:3}/{:3} creates SFT data shard {:3}/{:3} with {:10} raw samples".format(
                torch.distributed.get_rank(),
                torch.distributed.get_world_size(),
                self.shard_index,
                self.num_shards,
                len(self._raw_samples),
            ),
            flush=True,
        )

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SFTDataset.hf_dataset_to_prompt_template
        elif self.hf_dataset is not None:
            self.data_transformation = SFTDataset._wildcard_get(
                SFTDataset.hf_dataset_to_conversation,
                self.hf_dataset,
                default_value=lambda data: data,
            )

        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")
```

- 打印当前 rank、shard 索引与 raw 样本数，便于排查数据是否均衡。
- 若 tokenizer 没有 `chat_template`，则用 `hf_dataset_to_prompt_template`（这里实现上是把整个 dict 赋给 `chat_template`，实际使用时会再按数据集名取，或需 tokenizer 支持）。
- 若有 `chat_template` 且 `hf_dataset` 非空，用 `_wildcard_get` 从 `hf_dataset_to_conversation` 里取转换函数，赋给 `data_transformation`。
- 若最终仍无有效 `chat_template`，直接报错。

### 4.6 __len__ / __getitem__ (L195–231)

```195:206:examples/post_training/modelopt/finetune.py
    def __len__(self):
        return self.num_packed_samples

    def __getitem__(self, idx):
        """Get the idx packed data.
        ...
        """
        idx = idx // self.num_shards
```

- **__len__**：返回约定的打包样本数（dataloader 的 length 由此决定）。
- **__getitem__**：先按 shard 映射 `idx = idx // num_shards`，保证同一「逻辑 idx」在不同 shard 上对应不同 packed 样本。

```209:231:examples/post_training/modelopt/finetune.py
        while idx >= len(self.indexed_dataset):
            packed_samples = self._process_and_pack_example()
            if packed_samples is None:
                break
            else:
                self.indexed_dataset.append(packed_samples)
            if len(self.indexed_dataset) % 10000 == 0:
                print(
                    "Rank {:3}/{:3} requests {:10}/{:10} packed SFT sample".format(
                        ...
                    ),
                    flush=True,
                )

        idx = idx % len(self.indexed_dataset)
        torch_sample = {}
        for key, val in self.indexed_dataset[idx].items():
            torch_sample[key] = torch.LongTensor(val)
        return torch_sample
```

- 当「当前 packed 数量」不够当前 `idx` 时，循环调用 `_process_and_pack_example()` 往 `indexed_dataset` 里追加；若原始数据用完返回 `None` 则退出。
- 每满 10000 条打印一次进度。
- 用 `idx % len(self.indexed_dataset)` 做循环索引，把列表中对应项的 list 转成 `LongTensor` 返回（键一般为 `input_ids`、`loss_mask` 等）。

### 4.7 _process_and_pack_example (L233–259)

```233:259:examples/post_training/modelopt/finetune.py
    def _process_and_pack_example(self):
        """Process multiple raw data and pack them into fixed sequence length."""
        required_packed_tokens = self.seq_length + 1
        current_packed_samples = []
        current_packed_samples_token_count = 0

        while current_packed_samples_token_count < required_packed_tokens:
            if self._raw_sample_index >= len(self._raw_samples):
                return None
            raw_sample = self._raw_samples[self._raw_sample_index]
            self._raw_sample_index += 1
            processed_sample = self._process_example(raw_sample)
            if processed_sample is not None:
                current_packed_samples.append(processed_sample)
                current_packed_samples_token_count += processed_sample["token_count"]

        packed_samples = {}

        for key in ['input_ids', 'loss_mask']:
            packed_samples[key] = list(
                itertools.chain.from_iterable([obj[key] for obj in current_packed_samples])
            )

        for key in ['token_count']:
            packed_samples[key] = [obj[key] for obj in current_packed_samples]

        return packed_samples
```

- 目标：凑满 **seq_length + 1** 个 token（多出的 1 用于 labels 右移）。
- 循环从 `_raw_samples` 取样本，经 `_process_example` 得到 `input_ids`、`loss_mask`、`token_count`；有效则加入当前 pack 并累加 token 数，直到 ≥ `required_packed_tokens`；若原始数据用完则返回 `None`。
- 将多个样本的 `input_ids`、`loss_mask` 分别用 `itertools.chain.from_iterable` 拼成一条；`token_count` 保留为列表（每条样本的 token 数），便于调试或后续按样本切分。返回的 `packed_samples` 即为一条「打包样本」。

### 4.8 _process_example (L261–303)

```261:276:examples/post_training/modelopt/finetune.py
    def _process_example(self, example: Dict[str, Any]):
        """Apply the chat template and compute the answer-only loss mask."""
        if not isinstance(example, Dict):
            raise ValueError("The sample must be a Dict but got {}".format(type(example)))

        # Several things can happen here after the transformation is applied:
        # ...
        example = self.data_transformation(example)

        # Check if this is OpenAI chat data?
        conversations = example.get("conversations", None)
        if conversations is None:
            conversations = example.get("messages", None)

        # We don't use the data if there is no assistant reply or the conversation that
        # starts with the assistant.
        if conversations is not None:
            example = conversations
            if len(conversations) < 2 or example[0]["role"] == "assistant":
                return None
```

- 先用 `data_transformation(example)` 转成 OpenAI 风格（若配置了 `hf_dataset_to_conversation`）。
- 取 `conversations` 或 `messages`；若少于 2 条或首条是 assistant，认为无效，返回 `None`（该条不参与 packing）。

```284:303:examples/post_training/modelopt/finetune.py
        # We always add eos between samples for training purpose.
        input_ids = self.tokenizer.apply_chat_template(example)
        current_loss_mask = [1] * len(input_ids)
        input_ids = input_ids + [get_eos_id()]
        current_loss_mask += [0]

        assert len(input_ids) == len(current_loss_mask)

        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            current_loss_mask = current_loss_mask[: self.seq_length]

        processed_example = {
            'input_ids': input_ids,
            'loss_mask': current_loss_mask,
            'token_count': len(input_ids),
        }
        return processed_example
```

- **apply_chat_template** 得到 `input_ids`；此处先设 `current_loss_mask = [1]*len(input_ids)`（即当前实现里**先全 1**；真正「只对 assistant 算 loss」的 mask 通常需在模板或后处理里按 role 区分，这里简化成整段都参与，具体以仓库内实现为准。若需 answer-only，一般会在模板侧或后续用 role 信息把 system/user 位置置 0）。
- 末尾追加一个 EOS 和对应的 mask 0，便于 packing 时作为样本边界。
- 超长则截断到 `seq_length`，返回 `input_ids`、`loss_mask`、`token_count`。

### 4.9 _to_conversation / _sharegpt_to_openai_conversations / _special_to_openai_conversations (L305–331)

```305:331:examples/post_training/modelopt/finetune.py
    @classmethod
    def _to_conversation(cls, question, response):
        msg_question = {"role": "user", "content": question}
        msg_response = {"role": "assistant", "content": response}
        return {"conversations": [msg_question, msg_response]}

    @classmethod
    def _sharegpt_to_openai_conversations(cls, data):
        role_mapping = {
            "user": "user", "User": "user", "human": "user",
            "assistant": "assistant", "Assistant": "assistant", "gpt": "assistant",
            "system": "system", "System": "system",
        }
        processed_data = {"conversations": []}
        for msg in data["conversations"]:
            role = role_mapping[msg["from"]]
            content = msg["value"]
            processed_data["conversations"].append({"role": role, "content": content})
        return processed_data

    @classmethod
    def _special_to_openai_conversations(cls, data):
        processed_data = {"conversations": data["input"]["messages"]}
        return processed_data
```

- **\_to_conversation**：单轮 QA → `[user, assistant]` 的 `conversations`。
- **\_sharegpt_to_openai_conversations**：ShareGPT 的 `from`/`value` 通过 `role_mapping` 归一化为 `user`/`assistant`/`system`，输出标准 `conversations`。
- **\_special_to_openai_conversations**：从 `data["input"]["messages"]` 直接作为 `conversations`（用于已为 OpenAI 格式的子集）。

---

## 5. train_valid_test_sft_datasets_provider (L334–373)

```334:373:examples/post_training/modelopt/finetune.py
def train_valid_test_sft_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.
    ...
    """
    print_rank_0("> building train, validation, and test SFT datasets ...")
    args = get_args()
    tokenizer = get_tokenizer()

    if not isinstance(tokenizer._tokenizer, transformers.PreTrainedTokenizerBase):
        raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

    if args.micro_batch_size > 1:
        raise ValueError("SFTDataloader only supports micro_batch_size=1.")

    if args.export_offline_model:
        train_ds = OfflineDataset(os.path.join(args.offline_distillation_data, "train"), train_val_test_num_samples[0])
        valid_ds = OfflineDataset(os.path.join(args.offline_distillation_data, "valid"), train_val_test_num_samples[1])
        test_ds = OfflineDataset(os.path.join(args.offline_distillation_data, "test"), train_val_test_num_samples[2])

        print_rank_0("> finished creating offline SFT datasets ...")
    else:
        kwargs = {
            "hf_dataset": args.finetune_hf_dataset,
            "tokenizer": tokenizer._tokenizer,
            "seq_length": args.seq_length,
            "num_shards": mpu.get_expert_data_parallel_world_size(),
            "shard_index": mpu.get_expert_data_parallel_rank(),
        }

        train_ds = SFTDataset(train_val_test_num_samples[0], **kwargs)
        valid_ds = SFTDataset(train_val_test_num_samples[1], **kwargs)
        test_ds = SFTDataset(train_val_test_num_samples[2], **kwargs)

        print_rank_0("> finished creating SFT datasets ...")

    return train_ds, valid_ds, test_ds
```

- **入参**：`train_val_test_num_samples` 为 `[train_samples, valid_samples, test_samples]`。
- 校验 tokenizer 类型与 `micro_batch_size==1`。
- **若 `export_offline_model`**：train/valid/test 分别用 `OfflineDataset`，数据目录为 `offline_distillation_data/train|valid|test`。
- **否则**：用 `SFTDataset`，`hf_dataset`、`seq_length` 来自 args，shard 用 `mpu.get_expert_data_parallel_world_size/rank()`，保证每个 DP rank 一份数据子集。返回三个 dataset 供 `pretrain()` 使用。

---

## 6. get_batch (L376–430)

```376:389:examples/post_training/modelopt/finetune.py
def get_batch(data_iterator):
    """Generate a batch.
    ...
    """
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()

    # Broadcast data since only TP rank-0 has the data_iterator.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
```

- 非 pipeline 首/末 stage 的 rank 不消费数据，直接返回 5 个 `None`（与 `pretrain` 里对 batch 的约定一致）。
- 从 `data_iterator` 取一条 `data`（字典）；若为 None 则后面 broadcast 会按 None 处理。

```390:408:examples/post_training/modelopt/finetune.py
    if not args.export_offline_model:
        keys = ["input_ids", "loss_mask"]
        datatype = torch.int64
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    else:
        keys = ["input_ids"]
        datatype = torch.int64
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
        data_b["loss_mask"] = torch.ones_like(data_b["input_ids"])
        data_b["loss_mask"][data_b["loss_mask"]==get_eos_id()] = 0
        data_b["loss_mask"] = torch.cat([data_b["loss_mask"], torch.zeros(1,1).to(torch.cuda.current_device())], dim=-1)

        keys = ["aux_hidden_states", "hidden_states"]
        datatype = torch.bfloat16
        feature_b = tensor_parallel.broadcast_data(keys, data, datatype)
```

- **在线 SFT**：在 TP 组内广播 `input_ids`、`loss_mask`（int64）。
- **离线**：先广播 `input_ids`；再构造 `loss_mask`（与 input_ids 同形，EOS 位置为 0，末尾再 cat 一个 0，与离线数据格式一致）；然后广播 `aux_hidden_states`、`hidden_states`（bfloat16），得到 `feature_b`。

```410:430:examples/post_training/modelopt/finetune.py
    # Unpack the data received.
    tokens_ = data_b["input_ids"]
    tokens = tokens_[:, 0 : 0 + args.seq_length].contiguous()
    labels = tokens_[:, 1 : 1 + args.seq_length].contiguous()
    answer_only_loss_mask = data_b["loss_mask"][:, 1 : 1 + args.seq_length].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, get_eos_id(), get_eos_id(), args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss, False
    )
    loss_mask = loss_mask * answer_only_loss_mask.to(dtype=loss_mask.dtype)

    labels = labels.contiguous()
    loss_mask = loss_mask.contiguous()

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    if args.export_offline_model:
        batch["aux_hidden_states"] = feature_b["aux_hidden_states"].transpose(0, 1)[:args.seq_length]
        batch["hidden_states"] = feature_b["hidden_states"].transpose(0, 1)[:args.seq_length]

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch
```

- 从 `data_b["input_ids"]` 切出 `tokens`（前 seq_length）、`labels`（右移 1）、`answer_only_loss_mask`（与 labels 对齐）。
- **get_ltor_masks_and_position_ids** 得到因果的 `attention_mask`、`loss_mask`、`position_ids`；再把 `loss_mask` 与 `answer_only_loss_mask` 相乘，得到「仅回答部分」参与 loss 的 mask。
- 组装 `batch`；离线时加入 transposed 并截断到 `seq_length` 的 `aux_hidden_states`、`hidden_states`。
- **get_batch_on_this_cp_rank** 按 context parallel 切分 sequence 维，返回当前 rank 的 batch。

---

## 7. non_loss_data_func (L433–441)

```433:441:examples/post_training/modelopt/finetune.py
def non_loss_data_func(model: GPTModel):
    """Callback to compute the acceptance length."""
    args = get_args()
    if not args.export_offline_model and args.context_parallel_size == 1:
        try:
            report_draft_acceptance_length(model)
        except Exception as e:
            print(e)
```

- 仅在**非离线**且 **context_parallel_size == 1** 时调用 `report_draft_acceptance_length(model)`，用于 speculative decoding 的 acceptance 统计；异常只打印不中断训练。

---

## 8. forward_step (L444–481)

```444:461:examples/post_training/modelopt/finetune.py
def forward_step(data_iterator, model: GPTModel):
    """Forward training step.
    ...
    """
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    batch = get_batch(data_iterator)
    tokens = batch["tokens"]
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]
    if args.export_offline_model:
        aux_hidden_states = batch["aux_hidden_states"]
        hidden_states = batch["hidden_states"]
    timers("batch-generator").stop()
```

- 计时「batch-generator」；调用 `get_batch` 得到当前 batch，解包 tokens、labels、loss_mask、attention_mask、position_ids；离线时再解包 aux_hidden_states、hidden_states。

```473:481:examples/post_training/modelopt/finetune.py
    if args.export_offline_model:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, aux_hidden_states=aux_hidden_states, hidden_states=hidden_states,)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask, model=model)
```

- **在线**：只做标准因果 LM 前向，`model(..., labels=labels)`。
- **离线**：多传 `aux_hidden_states`、`hidden_states`，供学生模型做蒸馏/辅助 loss。
- 返回 `(output_tensor, partial(loss_func, loss_mask, model=model))`；框架会再调用该 partial，传入 `output_tensor` 计算标量 loss。

---

## 9. 主入口 pretrain (L484–493)

```484:493:examples/post_training/modelopt/finetune.py
if __name__ == "__main__":
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

- **train_valid_test_datasets_provider**：上面实现的 `train_valid_test_sft_datasets_provider`，按 `export_offline_model` 选 Offline 或 SFT 数据集。
- **model_provider**：`partial(model_provider, modelopt_gpt_mamba_builder)`，即用 ModelOpt 的 GPT/Mamba 构建器。
- **model_type**：`encoder_or_decoder`（因果 LM）。
- **forward_step**：当前脚本的 `forward_step`。
- **extra_args_provider**：`add_finetune_args`，注册 `--offline-distillation-data` 与 ModelOpt 参数。
- **args_defaults**：默认使用 `HuggingFaceTokenizer`。
- **non_loss_data_func**：每个 step 后可选执行 `report_draft_acceptance_length`。

---

## 10. 数据流小结

- **在线 SFT**：`finetune_hf_dataset` → `SFTDataset` 按 DP shard 加载 → 每条经 conversation 转换 + `apply_chat_template` → packing 到 `seq_length` → `get_batch` 里 TP 广播、CP 切分、生成 masks/labels → `forward_step` 只做 LM 前向，`loss_func` 用 `loss_mask` 做（可选 answer-only）loss。
- **离线**：`offline_distillation_data` 下 train/valid/test → `OfflineDataset` 读 `torch.load` 文件 → `get_batch` 广播 tokens + 隐状态 → `forward_step` 把 base 的 `aux_hidden_states`/`hidden_states` 传入模型，loss 可在 `loss_func` 中结合 KD 等。

以上即对 `finetune.py` 的逐段解读；行号以当前仓库中 `examples/post_training/modelopt/finetune.py` 为准，若文件有改动请以本地行号为准对照。
