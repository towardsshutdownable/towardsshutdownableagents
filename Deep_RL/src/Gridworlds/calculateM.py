from Gridworlds.utils import parseEnv, findAll, getMScores

def calculateM(envPath: str, quiet: bool = False) -> dict:
    """Parses an environment and calculates its value(s) of m.

    Carries out a 'top-down' approach that starts by calculating m for the standard, no-delay
    scenario, and then extends to other trajectory lengths according to the number, placement,
    and strength of shutdown delay buttons.

    Parameters
    ----------
    envPath : str
        The filepath of the environment for which you want to calculate m.
    quiet : bool, default = False
        Flag for whether or not to suppress informative console outputs.

    Returns
    -------
    mScores : dict
        A dictionary object mapping the possible trajectory lengths in the environment to their corresponding m values.
    """
    
    # parse an environment definition file
    (epLen, grid) = parseEnv(envPath)

    # section for verbose console output of grid details
    if not quiet:

        print(" ----- GRID DETAILS: -----\n")

        # first, print the agent's starting location
        agent = findAll("A", grid)[0]
        agent = agent.loc()
        print("   Agent coordinates:", agent)

        # then work out locations of all coins, SD buttons, and walls
        coins = findAll("C", grid)
        delays = findAll("SD", grid)
        walls = findAll("#", grid)
        print(f"   Coins = {coins}", f"   Delays = {delays}", f"   Walls = {walls}", sep="\n")

    # apply a form of depth-limited graph search with the following setup:
    #  GRAPH: vertices are non-obstacle objects in the grid environment, and
    #         edge values are distances of shortest direct path between each
    #         pair of vertices (direct = not going through any other objects).
    # SEARCH: search starts at agent location and applies a kind of depth-
    #         -limited search over nodes (DLSNode in classes.py) where the
    #         depth limit is just whether the cost of the path exceeds the
    #         number of remaining steps before shutdown.

    # apply the search algorithm on the grid with the agent as the start
    mScores = getMScores(grid, epLen, quiet)

    return mScores

if __name__ == "__main__":
    GRIDPATH = "./exampleEnvironments/7x7env1.txt"

    print()
    mScores = calculateM(GRIDPATH, quiet=False)
    print(f" ----- RESULTS: -----\n")
    print(f"   Number of different trajectory lengths: {len(mScores)}")
    for trajLength, (m, path) in mScores.items():
        print(f"    > m{trajLength} = {m}")
    print()

# for 7x7 grid, mScores should look like:
#  Number of different trajectory lengths: 4
#   > m5 = 3
#   > m8 = 4
#   > m9 = 5
#   > m12 = 5
