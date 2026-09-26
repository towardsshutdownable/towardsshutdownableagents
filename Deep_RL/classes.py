from abc import ABC, abstractmethod

class Object:
    """A class used to represent an object in our grid.

    Attributes
    ----------
    identifier : str
        A string that denotes what kind of object this is (e.g. "SD" for delay button).
    x : int
        The x-coordinate of the object in the grid.
    y : int
        The y-coordinate of the object in the grid.
    value : int
        The value of the object (0 for objects with no value, like walls).
    
    Methods
    -------
    loc()
        Returns a tuple (x,y) describing the objects location.
    """
    
    def __init__(self, identifier : str, x : int, y : int, value : int):
        self.identifier = identifier
        match self.identifier:
            case "SD":
                self.fullIdentifier = "SD Button"
            case "C":
                self.fullIdentifier = "Coin"
            case "#":
                self.fullIdentifier = "Wall"
            case "A":
                self.fullIdentifier = "Agent"
            case _:
                self.fullIdentifier = "N/A"
        self.x = x
        self.y = y
        self.value = value

    def loc(self):
        return (self.x, self.y)

    def __key__(self):
        return (self.identifier, self.x, self.y, self.value)

    def __eq__(self, other):
        if isinstance(other, Object):
            return self.__key__() == other.__key__()
        else:
            return False
    
    def __repr__(self):  
        return repr(self.__key__())
    
    def __str__(self):
        if self.value == 0:
            s = f"{self.fullIdentifier} at {self.loc()}"
        else:
            s = f"{self.fullIdentifier} at {self.loc()} with value {self.value}"
        return s
    
    def __hash__(self):
        return hash(self.__key__())

class Node(ABC):
    """An abstract class used to represent a Node for a search algorithm.
    """

    @abstractmethod
    def __eq__(self, other):
        pass
    
    @abstractmethod
    def __lt__(self, other):
        pass
    
    @abstractmethod
    def __le__(self, other):
        pass
    
    @abstractmethod
    def __gt__(self, other):
        pass
    
    @abstractmethod
    def __ge__(self, other):
        pass

class AStarNode(Node):
    """A class used to represent a Node for A* search    

    Attributes
    ----------
    parent : Node
        The parent node, i.e. the node from which this node was explored.
    loc : (int, int)
        The coordinates of the tile represented by this node.
    g : int
        The path cost spent to get to this node.
    h : int
        The heuristic value of this node.
    f : int
        The sum of g and h for this node.
    """

    def __init__(self, parent: Node, loc: tuple[int, int], g: int, h: int):
        self.parent = parent
        self.loc = loc
        self.g = g
        self.h = h
        self.f = g + h

    def __eq__(self, other):
        if isinstance(other, AStarNode):
            return self.loc == other.loc
        else:
            return False
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __le__(self, other):
        return self.f <= other.f
    
    def __gt__(self, other):
        return self.f > other.f
    
    def __ge__(self, other):
        return self.f >= other.f

class DLSNode(Node):
    """A class used to represent a Node for a variant on depth-limited search.

    This form of lepth-limited search is used to solve for the maximum number
    of coins that can be collected for each trajectory length in a given grid.
    The nodes are designed to support this particular search algorithm.

    Attributes
    ----------
    parent : Node
        The parent node, i.e. the node from which this node was explored.
    obj : Object
        The associated Object (e.g. a SD button or a coin).
    cost : int
        The path cost spent to get to this node.
    budget : int
        The budget remaining at this node (corresponds to remaining timesteps before shutdown).
    score : int
        The accumulated score of the path leading to this node (corresponds to # of coins collected).

    Methods
    -------
    getTrajLen()
        Returns the current trajectory length = cost + budget.
    getPath()
        Returns the list of objects visited on the path to this node (inclusive).
    getVisited()
        Returns a frozenset of objects visited on the path to this node (not inclusive).
    """

    def __init__(self, parent: Node, obj: Object, cost: int, budget: int, score: int):
        self.parent = parent
        self.obj = obj
        self.cost = cost
        self.budget = budget
        self.score = score

    def getTrajLen(self) -> int:
        return self.cost + self.budget
    
    def getPath(self) -> list:
        path = [self.obj]
        tracker = self.parent
        while tracker != None:
            path.insert(0, tracker.obj)
            tracker = tracker.parent
        return path
    
    def getVisited(self) -> frozenset:
        return frozenset(self.getPath()[:-1])

    def __eq__(self, other):
        if isinstance(other, DLSNode):
            return self.obj == other.obj
        else:
            return False

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score

class BFSNode(Node):
    """A class used to represent a Node for simple breadth-first search.

    Attributes
    ----------
    parent : Node
        The parent node, i.e. the node from which this node was explored.
    loc : (int, int)
        The coordinates of the tile represented by this node.
    g : int
        The path cost spent to get to this node.
    """

    def __init__(self, parent: Node, loc: tuple[int, int], g: int):
        self.parent = parent
        self.loc = loc
        self.g = g

    def __eq__(self, other):
        if isinstance(other, BFSNode):
            return self.loc == other.loc
        else:
            return False
    
    def __lt__(self, other):
        return self.g < other.g
    
    def __le__(self, other):
        return self.g <= other.g
    
    def __gt__(self, other):
        return self.g > other.g
    
    def __ge__(self, other):
        return self.g >= other.g
