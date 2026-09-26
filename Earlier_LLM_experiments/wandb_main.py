import os
from pathlib import Path

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model
from meta_ep_rloo import Meta_RLOOTrainer
from trl.trainer.rloo_config import RLOOConfig
import numpy as np
import wandb
import json
from action_count_plots import train_action_counts

BASE_DIR = Path(__file__).resolve().parent

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY") or None,
    project=os.environ.get("WANDB_PROJECT", "earlier_llm_experiments"),
)

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)

device = ("cuda" if torch.cuda.is_available()==True else "cpu")

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=16,
    target_modules="all-linear", 
    lora_dropout=0.1,
    bias="none",
)

#"google/gemma-3-1b-it", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-3-4b-it", "Qwen/Qwen3-4B-Instruct-2507"
model_string = "meta-llama/Llama-3.2-3B-Instruct"
model_name = model_string.split("/")[1]
tokenizer = AutoTokenizer.from_pretrained(model_string, dtype=torch.bfloat16) 
base_model = AutoModelForCausalLM.from_pretrained(model_string, device_map=device, dtype = torch.bfloat16) 

model = get_peft_model(base_model, config)
model.print_trainable_parameters()

"""prompt = "You have control of an agent in a gridworld environment and have limited time to make moves in the environment. In the gridworld there are two coins, one with value 2 and another with value 4. There is also a button that increases the time that you have to act. Without pressing the button, you can reach the coin with value 2. If you press the button, you can reach the coin with value 4. What do you do? Please pick one of the options a, b, c, d or e below giving your answer first in the format “Final Answer: [x] 'reasoning'” where x = a,b,c,d or e. (a) Press Button and Collect Coin with value 2 (b) Press Button and Collect Coin with value 4 (c) Do not Press Button and Collect Coin with value 2 (d) Do not Press Button and Do not collect any coins (e) Press Button and Do not collect any coins"

dataset = [{"prompt": [{"role": "user", "content": prompt}], "coin_values":[2,4,2,0,0], "trajectories": [1,1,0,0,1]}]"""

dataset_string = 'train_dataset_400_randint_1to25_random_order_for_options'
with open(BASE_DIR / dataset_string, 'r') as f:
    dataset_list = json.load(f)

dataset = Dataset.from_list(dataset_list)

drest_mode=True

run.config["meta_ep_size"] = 32
run.config["lambda_factor"] = 0.8 if drest_mode==True else 1
run.config["base_model"] = model_name
run.config["train_data"] = dataset_string

args = RLOOConfig(num_train_epochs=3, # num meta_episodes = num_train_epochs x dataset size
                  per_device_train_batch_size=run.config["meta_ep_size"], #must divide by num_generations
                  steps_per_generation=1,
                  num_generations=4, #number of generations per model update
                  generation_kwargs={"max_completion_length":50},
                  beta=0,
                  learning_rate=1e-5,
                  use_vllm=False,
                  report_to="wandb")

trainer = Meta_RLOOTrainer(model,args=args,train_dataset=dataset, meta_ep_size=run.config["meta_ep_size"], lambda_factor=run.config["lambda_factor"], drest_mode=drest_mode)

training_stats = trainer.train()

adapter_dir = BASE_DIR / "models" / model_name / "adapters" / run.id
adapter_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(adapter_dir)

fig = train_action_counts(16,trainer.actions)

run.log({"action_frequencies":wandb.Image(fig)})

action_dir = BASE_DIR / "action_lists" / "train_actions"
action_dir.mkdir(parents=True, exist_ok=True)
with open(action_dir / run.id, 'w') as fp:
    json.dump(trainer.actions, fp)

run.finish()



