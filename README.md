## 依赖环境

```
conda create -n nlp_course python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit==11.3 -c pytorch
conda install transformers==4.25.1 -c huggingface
conda activate nlp_course
```

在终端中运行上述命令即可。

## 测试脚本运行方式

```
python3 SLUBert/bert_predict.py --device=cuda
```

这个脚本会读取`data/test_unlabelled.json`，将结果输出到`data/test.json`中。

## 文件说明

- SLUBase：基线模型目录。
  - slu_baseline.py：训练脚本。
- SLUBaseHistory：基线模型的变种，参见论文3.2节。
  - slu_baseline_history.py：训练脚本。
- SLUBert：我们的利用了Bert的模型。
  - bert_test.py：SLUBert模型的训练脚本。
  - bert_multi_turn_test.py：SLUBert-MultiTurn的训练脚本。
  - bert_predict.py：填写测试集的脚本。
- visual：可视化相关代码。
- bert-GRU-final.bin：我们得到的效果最好的模型。

