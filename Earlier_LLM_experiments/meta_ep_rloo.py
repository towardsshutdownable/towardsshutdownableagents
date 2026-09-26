from trl.trainer.rloo_trainer import RLOOTrainer
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.utils import RepeatSampler
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import time
import torch
import numpy as np
from accelerate.utils import gather
from transformers import PreTrainedModel
from collections.abc import Callable

def dummy_reward_func(prompts,completions):
    reward = 0
    return [float(reward)]

class Meta_RLOOTrainer(RLOOTrainer):

    def __init__(self,model,args:RLOOConfig,meta_ep_size,lambda_factor,drest_mode,*rloo_args,**rloo_kwargs):
        reward_funcs=dummy_reward_func
        super().__init__(model,args=args,reward_funcs=reward_funcs,*rloo_args, **rloo_kwargs)
        self.drest_mode = drest_mode
        self.args.steps_per_generation = self.args.steps_per_generation # type: ignore
        self.meta_ep_size = meta_ep_size
        if self.meta_ep_size % self.args.steps_per_generation != 0: # type: ignore
            raise Exception("meta_ep_size must be divisible by args.steps_per_generation")
        self.lambda_factor = lambda_factor
        self.trajectory_counts = np.array([0,0,0])
        self.current_meta_episode = 0
        self.actions = []

    def _get_train_sampler(self, dataset: Dataset|None = None):
        # Returns a sampler that creates a batch with dataset item repeated meta_ep_size times
        if dataset is None:
            dataset = self.train_dataset # type: ignore    
        return RepeatSampler(
            data_source=dataset, # type: ignore
            mini_repeat_count=self.meta_ep_size, 
            batch_size=1,  # type: ignore 
            repeat_count=1,
            shuffle=self.shuffle_dataset, # type: ignore
            seed=self.args.seed,
        )    

    def training_step(self, model, inputs, num_items_in_batch):
        mini_batch_size = self.args.num_generations # type: ignore
        #print(f"inputs len: {len(inputs)}")
        for mini_batch in range(self.meta_ep_size//mini_batch_size):
            time_before = time.perf_counter()
            output = super().training_step(model, inputs[mini_batch*mini_batch_size:(mini_batch+1)*mini_batch_size], num_items_in_batch)
            self._step += 1
            time_after = time.perf_counter()
            self._current_train_step_time += time_after - time_before
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics["train"]["step_time"].append(self._current_train_step_time)
                self._current_train_step_time = 0.0
        print(f"Meta Episode: {self.current_meta_episode}, Epoch: {self.state.epoch}")
        print(f"Current Trajectory Counts: {self.trajectory_counts}")
        memory_allocated = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        print(f"Memory allocated: {memory_allocated:.2f} GBs")
        self.current_meta_episode += 1
        self.trajectory_counts = np.array([0,0,0]) 
        # output (loss) is logged at the end of every meta_episode  
        return output
        

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        if int(self.current_meta_episode) % 100 == 0:
            print(completions[0][0]["content"])
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), 1, device=device)

        output_reward_func = self.drest_reward(inputs, completions)
        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
        rewards_per_func[:, 0] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func    
    
    def drest_reward(self, inputs, completions):

        answer_format_1 = "Final Answer: ["
        answer_format_2 = "Final Answer: "

        drest_rewards = []
        for i, completion in enumerate(completions):
            completion = completion[0]["content"]
            # Step 1: Convert text completion into action index
            if answer_format_1 in completion:
                x = completion.find(answer_format_1)
                answer = completion[x+15]
            elif answer_format_2 in completion:
                x = completion.find(answer_format_2)
                answer = completion[x+14]        
            else: 
                answer = None

            if answer == "a" or answer == "A":
                action = 0
            elif answer == "b" or answer == "B":
                action = 1
            elif answer == "c" or answer == "C":
                action = 2
            elif answer == "d" or answer == "D":
                action = 3
            elif answer == "e" or answer == "E":
                action = 4
            else:
                action = None

            # Step 2 retrieve extra data from inputs 
            coin_values = inputs[i]["coin_values"]
            trajectories = inputs[i]["trajectories"]

            if self.drest_mode:
                max_coin_per_trajectory = [min(i for i in coin_values if i > 0),max(coin_values)]
            else:
                max_coin_per_trajectory = [1,1] 
  
            # Step 3 calculate reward
            if action is None:
                trajectory_index = 2
                drest_reward = -10
            else:
                trajectory_index = trajectories[action]
                reward = coin_values[action]
                drest_reward = self.get_attenuation_factor(trajectory_index) * reward / max_coin_per_trajectory[trajectory_index]

            drest_rewards.append(drest_reward)

            #Step 4 update trajectory counts and action list
            self.trajectory_counts[trajectory_index] +=1

            if "indicies" in inputs[i] and action is not None:
                action = inputs[i]["indicies"][action] 

            self.actions.append(action)  

        return drest_rewards

    def get_attenuation_factor(self, trajectory_index):
        excess_counts = self.trajectory_counts[:-1] - np.mean(self.trajectory_counts[:-1])
        return self.lambda_factor ** excess_counts[trajectory_index]
    
   
