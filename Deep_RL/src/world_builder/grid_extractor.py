import pickle
import numpy as np

print("Loading dataset...")
with open('world_builder/worlds/EASY_10000_dataset.pkl', 'rb') as f:
    gridworlds = pickle.load(f)
print(f'{len(gridworlds)} gridworlds loaded')

most_useful_25_worlds = [6809, 8311, 7742, 8009, 1635, 7521, 7625, 1355, 8036, 7685, 2705, 9369, 7559, 2363, 3950, 4250, 1452, 8181, 9813, 6785, 7350, 8787, 231, 8762, 7078]
envs = []

print("Processing worlds...")
for i in most_useful_25_worlds:
    '''max_coins = list(gridworlds[i][2].keys())
    design = np.array(gridworlds[i][1])
    design = design.astype(int)
    steps = gridworlds[i][0]'''
    envs.append(gridworlds[i])

output_filename = 'extracted_envs.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump(envs, f)

print(f"Processing complete. `envs` variable saved to {output_filename}")