# nlpdl_project
Final project repository for NLPDL-2022fall.
## datasets
数据处理的中间结果。
网盘链接：https://disk.pku.edu.cn:443/link/17F56BA35DF9EEE9AC10709F148921A5
## models
- naive_post_training：naive post training模型的checkpoint。
- bioasq
  - pretrain
    bioasq_improve_naive：在naive post training基础上的针对BioASQ的TAPT预训练模型。
  - bioasq_naive：使用naive post training作为预训练模型的fine-tune模型。
  - bioasq_tapt：使用bioasq的TAPT作为预训练模型的fine-tune模型。
  - bioasq_naive：使用基于naive post training的bioasq的TAPT作为预训练模型的fine-tune模型。
- chemprot
  - pretrain
    chemprot_improve_naive：在naive post training基础上的针对CHEMPROT的TAPT预训练模型。
  - chemprot_naive：使用naive post training作为预训练模型的fine-tune模型。
  - chemprot_tapt：使用chemprot的TAPT作为预训练模型的fine-tune模型。
  - chemprot_naive：使用基于naive post training的chemprot的TAPT作为预训练模型的fine-tune模型。
网盘链接：https://disk.pku.edu.cn:443/link/E423A3DB43979FA6CD54BD29400B13EA
## files
- data_prepare.py：用于VAMPIRE的数据准备。
- add_data.py：用于SBERT的数据准备。
- run_mlm.py：post training代码。
- main.py：fine-tune代码。
