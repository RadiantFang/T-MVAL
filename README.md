# T-MVAL

用于生成并优化长度为 `200bp` 的 DNA 序列。当前主流程由 `Oracle -> Diffusion -> MHVN warm-up -> Reward-Guided Finetuning -> Sampling` 组成。

## 主入口脚本

- `train_oracle.py`
- `prepare_top_hepg2_data.py`
- `diffusion_pretraining.py`
- `value_network_warmup.py`
- `finetuning.py`
- `sample_finetuned_diffusion.py`

## 数据来源

- 论文：`Machine-guided design of cell-type-targeting cis-regulatory elements`
- 数据：<https://zenodo.org/records/10698014/files/DATA-Table_S2__MPRA_dataset.txt?download=1>

## 最小运行

```bash
python3 train_oracle.py
python3 prepare_top_hepg2_data.py
python3 diffusion_pretraining.py
python3 value_network_warmup.py
python3 finetuning.py
```

采样：

```bash
python3 sample_finetuned_diffusion.py -f 40
```

## 说明

- 模型权重地址：<https://huggingface.co/FmikGy/T-MVAL>，正在陆续上传
- `value_network_warmup.py` 默认会额外导出 `checkpoints/value_replay_seed.pt`
- `finetuning.py` 默认使用 `oracle_finetune.ckpt` 作为训练 oracle，`oracle_eval.ckpt` 作为评估 oracle
- `enformer_regressor.py` 使用自定义 `load_from_checkpoint()` 来兼容当前仓库的 `.ckpt` 文件
