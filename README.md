# Mistral-finetune

<a target="_blank" href="https://colab.research.google.com/github/mistralai/mistral-finetune/blob/main/tutorials/mistral_finetune_7b.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


`mistral-finetune` is a light-weight codebase that enables memory-efficient and performant finetuning of Mistral's models.
It is based on [LoRA](https://arxiv.org/abs/2106.09685), a training paradigm where most weights are frozen and only 1-2% of additional weights in the form of low-rank matrix perturbations are trained. 

This repo is a fork of the original `mistral-finetune` ](https://arxiv.org/abs/2106.09685) adapted to the training of models dedicated to ICD-10 coding from clinical notes.

The purpurse of this fork is to help information medical teams to finetune Mistral model on the ICD-10 coding task (in french) with so called annotated data :
- data = clinical notes (1 note or the concatenation of all the notes available for the patient in EMR). The restriction is that the model can only take a fixed number of token as entry.
- annotation = ICD-10 codes of the PMSI resume. 2 formats are possible
  * when using classification : lits of code (ex [C509, I10,...])
  * when using generative model : definition of the code (code) (ex : Hypertention artérielle primitive (I10)).

For this finetuning a generative model we can use 2 paradigm :
- Next token prediction : you give the model a long text, and the task is to prodict next word. For ICD-10 coding, we train the model will on a text which is the concatenation of the note and ICD-10 coding. This task will princilally help the model to learn contextualised reprensentations of medical words of the clinical note and of the of ICD-10 definitions and codes/ 
- Instruction prediction : the model here is seen as an assistant. You give to the assistant a context (medical ICD-10 coding from clinical notes) and a question (what codes will you choose for the following clinical note), and the assistant will give a correct answer (the ICD-10 codes).

## Installation

To get started with Mistral LoRA fine-tuning, follow these steps:

1. Clone this repository:
```
cd $HOME && git clone https://github.com/24p11/recode-with-mistral-finetune.git
```

2. Install all required dependencies:
```
cd mistral-finetune
pip install -r requirements.txt
```

## Model download

We recommend fine-tuning one of the 7B Instruct v3 Mistral models which you can download from the official Mistral repo :
7B Instruct v3 | [7B Instruct v3](https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar) 
or from hugginface see 

E.g., to download the 7B-base model you can run the following command:
```sh
mkdir -p ~/${HOME}/mistral_models
cd ${HOME} && wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar
tar -xf mistral-7B-v0.3.tar -C mistral_models
```

Make sure to modify your training script and add the path to the downloaded 
folder as `model_id_or_path`.

E.g., modify [example/7B.yaml](https://github.com/mistralai/mistral-finetune/blob/main/example/7B.yaml) to include the absolute path to `$HOME/mistral_models/7B`:

```
model_id_or_path: "/Users/johndoe/mistral_models/7B"
```

## Prepare dataset 

To ensure effective training, `mistral-finetune` has strict 
requirements for how the training data has to be formatted.

All data files must be stored in jsonl format files.

You can build two types of data files:

### _Pretrain_:

Pretrain data corresponds to plain text data stored in the `"text"` key. E.g:

```jsonl
{"text": "Text contained in clinical note n°1. ICD-10 codes : definition 1 (code 1),..."}
{"text": "Text contained in clinical note n°2. ICD-10 codes : definition 1 (code 1),..."}
```

In the pretrain paradigm models are funetune with the next token prediction task.
### _Instruct_:

Currently two different types of instruction following data are supported:

- _Instruct_: In the conversational data stored in the `"messages"` key in the form of a list. Each list item is a dictionary containing the `"content"` and `"role"` keys. `"role"` is a string being one of 
  * "system" :  task contextualization
  * "user" : question the assistant will answer
  * "assistant" : expected result from the assistant
The loss will only be computed if "role" == "assistant". 

For ICD-10 coding in French we have adopted the following conventions :
- system : Vous êtes un modèle de langage en française spécialisé dans le codage des diagnostics selon la classification internationale des maladies version 10 (CIM-10) pour les résumés standardisés de sortie du programme de médicalisation des systèmes d'information français (PMSI). A partir des comptes rendus d'hospitalisation vous donnerez les codes diagnostics CIM-10 que l'on peut retenir pour le séjours en distiguant diagnostic principal, diagnostic relié et diagnostics associés.
- user : Générez le codage CIM-10 du résumé strandisé de sortie PMSI à partir du compte rendu d'hospitalisation suivant : texte du compte rendu
- assistant : Codes CIM 10 retenus pour le résumé strandisé de sortie PMSI : diagnostic principal : définition du code (code), diagnistic relié : aucun, diagnostic associé : définition diagnostic 1 (code 1),...
```jsonl
{
  "messages": [
     {
      "role": "system",
      "content": "Vous êtes un modèle de langage en française spécialisé dans le codage des diagnostics selon la classification internationale des maladies version 10 (CIM-10) pour les résumés standardisés de sortie du programme de médicalisation des systèmes d'information français (PMSI). A partir des comptes rendus d'hospitalisation vous donnerez les codes diagnostics CIM-10 que l'on peut retenir pour le séjours en distiguant diagnostic principal, diagnostic relié et diagnostics associés."
    },
    {
      "role": "user",
      "content": "Générez le codage CIM-10 du résumé strandisé de sortie PMSI à partir du compte rendu d'hospitalisation suivant : texte du compte rendu n°1"
    },
    {
      "role": "assistant",
      "content": "Codes CIM 10 retenus pour le résumé strandisé de sortie PMSI : diagnostic principal : définition du code (code), diagnistic relié : aucun, diagnostic associé : définition diagnostic 1 (code 1),..."
    }
  ]
}
{
  "messages": [
     {
      "role": "system",
      "content": "Vous êtes un modèle de langage en française spécialisé dans le codage des diagnostics selon la classification internationale des maladies version 10 (CIM-10) pour les résumés standardisés de sortie du programme de médicalisation des systèmes d'information français (PMSI). A partir des comptes rendus d'hospitalisation vous donnerez les codes diagnostics CIM-10 que l'on peut retenir pour le séjours en distiguant diagnostic principal, diagnostic relié et diagnostics associés."
    },
    {
      "role": "user",
      "content": "Générez le codage CIM-10 du résumé strandisé de sortie PMSI à partir du compte rendu d'hospitalisation suivant : texte du compte rendu n°2"
    },
    {
      "role": "assistant",
      "content": "Codes CIM 10 retenus pour le résumé strandisé de sortie PMSI : diagnostic principal : définition du code (code), diagnistic relié : aucun, diagnostic associé : définition diagnostic 1 (code 1),..."
    }
  ]
}
```

The notebook  [prepare_data_for_generative_finetuning](tutorials/prepare_data_for_generative_finetuning.ipynb) will show how to prepare data step by step. 

## Customizing training configuration

All the parameters of the training procedure are stored in yaml config file (see example/7B.yaml). Modify your training yaml to include the ultrachat dataset and verify the yaml

The example `mistral-finetune/examples/7B` defines reasonable parameters for learning rate, weight decay, etc... but you are advised to 
customize these settings for your use case.

Generally, a training configuration should fill the following parameters:

- `model_id_or_path` defines the model to start training from. This can be a path to a pre-trained model or a local model directory.
- `run_dir` defines the directory where training checkpoints and metrics are stored.
- `seq_len` defines the sequence length for training. This is the maximum length of input sequences the model will process. Samples are packed to reach a length of `seq_len` for maximum training efficiency.
- `batch_size` defines the number of training examples used per GPU. **Note**: The overall effective batch_size (in tokens) across all GPUs equals `num_gpus` x `batch_size` x `seq_len`.
- `max_steps` defines the maximum number of training steps. This is the total number of iterations the training process will run. It can be adjusted based on the specific needs of your training scenario. Total number of tokens seen during training is `max_steps` x `num_gpus` x `batch_size` x `seq_len`.
- `optim.lr` defines the learning rate. This is the initial learning rate for the optimizer.
- `optim.weight_decay` defines weight decay. Weight decay is a regularization technique used to prevent overfitting by penalizing large weights. We recommend leaving it at 0.1.
- `optim.pct_start` defines the percentage of the total training steps used for the learning rate warm-up phase before it starts to decrease. It corresponds to pct_start of PyTorch's OneCycleLR.
- `lora.rank` defines the size of the LoRA (Low-Rank Adaptation) adapters. We recommend 64 or less, which adjusts the rank of the low-rank decomposition used in LoRA.
- `seed` defines the random seed for initialization and data shuffling/sampling. Setting a seed ensures reproducibility of results.
- `log_freq` defines the logging frequency. This specifies how often (in steps) to log training metrics.
- `data.instruct_data` is the path to the instruction data used for training. This field has to be filled with one or multiple data sources in the format as explained above. Each data source should either be a path to a jsonl file or a path to a directory containing jsonl files followed by a weighting to define the importance of this dataset: `<path/to/data_source>:<weight>`. E.g.: `data.instruct_data: "/path/to/data1.jsonl:5.,/path/to/data2.jsonl:1.,/path/to/dir_of_jsonls:1."`
- `data.data` is an optional path to additional pretraining data in the format as explained above. Note that this field can be left blank.
- `data.eval_instruct_data` is an optional path to evaluation instruction data to run cross-validation at every `eval_freq` steps. Cross-validation metrics are displayed as `loss` and `perplexity`.
- `eval_freq` defines how often (in steps) to evaluate the model. This specifies the interval at which the model is evaluated on the validation set.
- `no_eval` is a flag to enable or disable intermediate evaluation. Setting it to False enables periodic evaluation during training.
- `ckpt_freq` defines how often (in steps) to save checkpoints. This specifies the interval at which the model's state is saved.
- `save_adapters` defines whether to only save the trained LoRA checkpoints or whether the trained LoRA should directly be merged into the base model and saved. **Note**: When setting `save_adapters=False` make sure that you have enough CPU and GPU memory to save the full model on a single process (this is usually only possible for the 7B model).
- `wandb.key` is used to pass your Weights & Biases (wandb) API key for logging. This allows you to log training metrics to the wandb dashboard.
- `wandb.project` defines the wandb project name. This is where the training run will be logged in the wandb interface.


## Verify dataset

Before starting a training run you should verify that your dataset is correctly formatted and get an 
estimation of the training time. You can do so by using the [./utils/validate_data](https://github.com/mistralai/mistral-finetune/blob/main/utils/validate_data.py) script.

Note that this step is crucial to ensure that the data is correctly formatted.

```
cd $HOME/mistral-finetune
python -m utils.validate_data --train_yaml example/7B.yaml
```

You should get a summary of the data input and training parameters:

```
Train States
 --------------------
{
   "expected": {
       "eta": "00:52:44",
       "data_tokens": 25169147,
       "train_tokens": 131072000,
       "epochs": "5.21",
       "max_steps": 500,
       "data_tokens_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "25169147.0"
       },
       "train_tokens_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "131072000.0"
       },
       "epochs_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "5.2"
       }
   },
}
```

Having `max_steps` set to 500 would lead to iterating through the dataset roughly 5 times which is reasonable, but might 
be a bit too much. A recommended setting is shown below which would only take 30min on a 8xH100 cluster.


## Start training

Having followed the [dataset verification section](#verify-dataset), we can now start training.
For faster training, we recommend setting max_steps to only 300. Make sure to define `run_dir` to your experiment folder and optionally set `wandb_project` to a Weights & Biases project for logging`, *e.g.*:
```
max_steps: 300
run_dir: "/Users/johndoe/ultra_chat_test"
wandb.project: ultra_chat
```
Save the training configuration and start training! Make sure to set `--nproc-per-node` to the number of available GPUs.

```
cd $HOME/mistral-finetune
torchrun --nproc-per-node 8 --master_port $RANDOM -m train example/7B.yaml
```

Training on ultra-chat should take around 30min on a 8xH100 node and the resulting weights should give an MT Bench score around 6.3.

Training on glaive should take around 1h on a 8xH100 node and the resulting weights should work nicely for function calling.


## Inference

Once your model is trained, you should try it out in inference. We recommend using [mistral-inference](https://github.com/mistralai/mistral-inference). 

Make sure to have `mistral_inference` correctly installed:
```
pip install mistral_inference
```

Assuming your `lora.safetensors` is saved under `$HOME/ultra_chat_test/checkpoints/checkpoint_000300/consolidated/lora.safetensors`, you can chat with the model using `mistral_inference`, *e.g.*:

```sh
mistral-chat /mnt/slow/runs/patrick/mistral-finetune/7B/ --max_tokens 256 --temperature 1.0 --instruct --lora_path $HOME/ultra_chat_test/checkpoints/checkpoint_000300/consolidated/lora.safetensors
```

## Model extension

**Important**: Note that one can only fine-tune mistral models that are compatible with the v3 tokenizer which entails that the models have a vocabulary size of 32768 - not 32000. One can however easily extend older version of vocabulary size 32000 to have a vocabulary size of 32768 by using:
```
python -m utils.extend_model_vocab --original_model_ckpt /folder/to/old/model --extended_model_ckpt /folder/to/extended/model
```

Once the extension has worked, one can fine-tune using the newly created model checkpoint in `/folder/to/extended/model`.

## FAQ:

> - What's the best practice of fine-tuning MoEs?

We see a higher degree of performance variance in when fine-tuning MoE models. It's not unusual to find that fine-tuning MoE models with different seeds can lead to a high variance in performance. We did not observe such a high variance with dense models. Therefore, we suggest running multiple instances of the same fine-tuning process on MoEs models and selecting the one that performs best.

> - How can I determine the number of tokens used during the model training process?
  
You can use the following script to find out: https://github.com/mistralai/mistral-finetune/blob/main/utils/validate_data.py. This script accepts a .yaml training file as input and returns the number of tokens the model is being trained on.

> - What should I do if I encounter a CUDA out-of-memory error?
  
One possible solution is to reduce the batch size per GPU. The batch size is equal to `seq_len` x `batch_size`. Try setting `batch_size` to 1 and reduce `seq_len`. You can define the `batch_size` and `seq_len` in the .yaml file.
