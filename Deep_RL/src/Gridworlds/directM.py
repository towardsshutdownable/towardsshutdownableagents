from Gridworlds.utils import parseEnv, findAll, getMScores

def directM(epLen: int, grid: list, quiet: bool = False) -> dict:
    """Takes a pre-parsed environment and calculates its value(s) of m.

    Carries out a 'top-down' approach that starts by calculating m for the standard, no-delay
    scenario, and then extends to other trajectory lengths according to the number, placement,
    and strength of shutdown delay buttons.

    Parameters
    ----------
    epLen : int
        The standard duration of the environment episode if no delay buttons are pressed.
    grid : list
        A 2D array containing the grid information in the format [<col1>, <col2>, etc.]
        ('A' for agent location,
         'C<v>' for a coin of value <v>,
         'SD<t>' for a shutdown button of time <t>,
         '#' for a wall, and '.' for an empty tile).
    quiet : bool, default = False
        Flag for whether or not to suppress informative console outputs.

    Returns
    -------
    mScores : dict
        A dictionary object mapping the possible trajectory lengths in the environment to their corresponding m values.
    """

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
    (epLen, grid) = parseEnv(GRIDPATH)

    print()
    mScores = directM(epLen, grid, quiet=False)
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
