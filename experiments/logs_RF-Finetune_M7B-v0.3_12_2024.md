Test Mistral 7B
- GPU : 1 V100
- BATCH SIZE :1
- MICROBACTH :8 (1 STEP = 8 encounters)
- SEQ LENGTH : 15000

Comments :
- Time per step : 1/2 min
- when LR increase, LOOS increase to nan value. see also : https://www.reddit.com/r/LocalLLaMA/comments/17dcyvg/is_this_loss_normal_qloramistral/
- Finaly this destroy the model which became not usable (see below)

```
2024-12-19 09:21:51 (CET) - 0:00:25 - train - INFO - TrainArgs: {'batch_size': 1,
 'checkpoint': True,
 'ckpt_freq': 1000,
 'data': {'data': '/export/home/cse170020/data/crh_omop_2024/generative_text/train.jsonl',
          'eval_instruct_data': '/export/home/cse170020/data/crh_omop_2024/generative_instruct/val.jsonl',
          'instruct': {'dynamic_chunk_fn_call': False, 'shuffle': False},
          'instruct_data': '/export/home/cse170020/data/crh_omop_2024/generative_instruct/train.jsonl',
          'shuffle': False},
 'eval_freq': 50,
 'log_freq': 1,
 'lora': {'dropout': 0.0, 'enable': True, 'rank': 64, 'scaling': 2.0},
 'max_norm': 1.0,
 'max_steps': 10000,
 'mlflow': {'experiment_name': None, 'tracking_uri': None},
 'model_id_or_path': '/export/home/cse170020/models/mistral_models/7B-v0.3',
 'no_ckpt': False,
 'no_eval': False,
 'num_ckpt_keep': 3,
 'num_microbatches': 8,
 'optim': {'lr': 0.0001, 'pct_start': 0.05, 'weight_decay': 0.1},
 'run_dir': '/export/home/cse170020/checkpoint/instruct_icd_v1',
 'save_adapters': True,
 'seed': 0,
 'seq_len': 15000,
 'wandb': {'key': None, 'offline': False, 'project': None, 'run_name': None},
 'world_size': 1}
2024-12-19 09:21:53 (CET) - 0:00:26 - finetune.wrapped_model - INFO - Reloading model from /export/home/cse170020/models/mistral_models/7B-v0.3/consolidated.safetensors ...
2024-12-19 09:21:56 (CET) - 0:00:29 - finetune.wrapped_model - INFO - Converting model to dtype torch.float16 ...
2024-12-19 09:22:50 (CET) - 0:01:24 - finetune.wrapped_model - INFO - Loaded model on cpu!
2024-12-19 09:22:50 (CET) - 0:01:24 - finetune.wrapped_model - INFO - Initializing lora layers ...
2024-12-19 09:22:51 (CET) - 0:01:25 - finetune.wrapped_model - INFO - Finished initialization!
2024-12-19 09:22:51 (CET) - 0:01:25 - finetune.wrapped_model - INFO - Sharding model over 1 GPUs ...
2024-12-19 09:22:55 (CET) - 0:01:29 - finetune.wrapped_model - INFO - Model sharded!
2024-12-19 09:22:55 (CET) - 0:01:29 - finetune.wrapped_model - INFO - 167,772,160 out of 7,415,795,712 parameters are finetuned (2.26%).
2024-12-19 09:22:56 (CET) - 0:01:29 - dataset - INFO - Lazily loading /export/home/cse170020/data/crh_omop_2024/generative_text/train.jsonl ...
2024-12-19 09:24:02 (CET) - 0:02:36 - dataset - INFO - Lazily loading /export/home/cse170020/data/crh_omop_2024/generative_instruct/train.jsonl ...

2024-12-19 09:24:38 (CET) - 0:03:12 - train - INFO - step: 000001 - done (%): 0.0 - loss: 1.724 - lr: 4.0e-06 - peak_alloc_mem (GB): 26.1 - alloc_mem (GB): 17.9 - words_per_second: 1167.7 - avg_words_per_second: 1167.7 - ETA: >2024-12-31 06:50:16
2024-12-19 09:26:15 (CET) - 0:04:48 - train - INFO - step: 000002 - done (%): 0.0 - loss: 1.682 - lr: 4.0e-06 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1247.4 - avg_words_per_second: 1206.2 - ETA: >2024-12-30 21:43:15
2024-12-19 09:27:53 (CET) - 0:06:27 - train - INFO - step: 000003 - done (%): 0.0 - loss: 1.667 - lr: 4.0e-06 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1215.3 - avg_words_per_second: 1209.3 - ETA: >2024-12-30 21:02:02
2024-12-19 09:29:36 (CET) - 0:08:09 - train - INFO - step: 000004 - done (%): 0.0 - loss: 1.709 - lr: 4.0e-06 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1172.5 - avg_words_per_second: 1199.9 - ETA: >2024-12-30 23:11:39
2024-12-19 09:31:14 (CET) - 0:09:47 - train - INFO - step: 000005 - done (%): 0.1 - loss: 1.704 - lr: 4.0e-06 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1224.5 - avg_words_per_second: 1204.7 - ETA: >2024-12-30 22:04:34

...

2024-12-20 01:06:01 (CET) - 15:44:35 - train - INFO - step: 000566 - done (%): 5.7 - loss: 8.025 - lr: 1.0e-04 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1207.7 - avg_words_per_second: 1201.9 - ETA: >2024-12-30 22:44:40
2024-12-20 01:07:40 (CET) - 15:46:13 - train - INFO - step: 000567 - done (%): 5.7 - loss: 7.902 - lr: 1.0e-04 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1217.9 - avg_words_per_second: 1201.9 - ETA: >2024-12-30 22:44:16
2024-12-20 01:09:14 (CET) - 15:47:47 - train - INFO - step: 000568 - done (%): 5.7 - loss: 7.935 - lr: 1.0e-04 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1278.4 - avg_words_per_second: 1202.0 - ETA: >2024-12-30 22:42:31
2024-12-20 01:10:48 (CET) - 15:49:21 - train - INFO - step: 000569 - done (%): 5.7 - loss: 8.132 - lr: 1.0e-04 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1277.2 - avg_words_per_second: 1202.2 - ETA: >2024-12-30 22:40:48
2024-12-20 01:12:27 (CET) - 15:51:01 - train - INFO - step: 000570 - done (%): 5.7 - loss: nan - lr: 1.0e-04 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1202.5 - avg_words_per_second: 1202.2 - ETA: >2024-12-30 22:40:48
2024-12-20 01:14:01 (CET) - 15:52:35 - train - INFO - step: 000571 - done (%): 5.7 - loss: nan - lr: 1.0e-04 - peak_alloc_mem (GB): 27.3 - alloc_mem (GB): 17.9 - words_per_second: 1277.0 - avg_words_per_second: 1202.3 - ETA: >2024-12-30 22:39:05
```
which have for consequence to broke the network... predictions after this training experience
```
Résultats de la prédiction pour le CRH:
 ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇
```
