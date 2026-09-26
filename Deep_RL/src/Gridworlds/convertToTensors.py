import pickle
import argparse
import os

import torch

from tqdm import tqdm

from Gridworlds.utils import parseEnv, gridArrayToTensor, getMScores

def parseGridToTensor(filePath: str) -> tuple[int, torch.Tensor, dict]:
    """Reads a gridworld from a text file and returns its state tensor.

    Converts from a text representation of a gridworld to a tensor representation
    with dimensions [numChannels, height, width]; the width and height are determined
    by the contents of the file, while the channels represent encodings of
    different objects in the gridworld (must be decided beforehand - see
    `utils.gridArrayToTensor` method for explanation of state tensor). Also packages
    information on possible trajectories / maximum available coins.

    Parameters
    ----------
    filePath : str
        The path to the gridworld to be parsed.

    Returns
    -------
    epLen : int
        The default number of timesteps before shutdown.
    gridTensor : torch.Tensor
        The state tensor that represents the input gridworld.
    mScores : dict
        A dictionary indexed by trajectory length containing the max score for that length
        and a path (list) of objects that gives you this max score.
    """

    # read the file and parse the gridworld
    epLen, grid = parseEnv(filePath)

    # convert the grid from list form to tensor form
    gridTensor = gridArrayToTensor(grid)

    # get the various trajectory lengths and maximum available coins for the grid
    mScores = getMScores(grid, epLen, quiet=True)

    return epLen, gridTensor, mScores

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, default="./generatedEnvironments/seed_10_EASY_x1000")

if __name__ == "__main__":

    args = parser.parse_args()

    datasetPath = args.data

    dataPath = os.path.join(datasetPath, "grids")

    dataset = []
    for fileName in tqdm(os.listdir(dataPath), "Converting to Tensors"):
        sample = parseGridToTensor(os.path.join(dataPath, fileName))
        dataset.append(sample)
    
    outputPath = os.path.join(dataPath, "..", "dataset.pickle")
    with open(outputPath, "wb") as out:
        pickle.dump(dataset, out)
