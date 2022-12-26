optimization_rules = "import os \n\
os.environ['XLA_USE_BF16']=\"1\" \n\
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000' \n"

# Straightforward implementation of the Singleton Pattern
# For counter class to get how many lines have been changed.
class ChangesCounter(object):
    _instance = None
    _count = 0
    def __new__(cls):
        if cls._instance is None:
            print('Creating the Counter object')
            cls._instance = super(ChangesCounter, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance

    def increment(self, val = 1):
        self._count = self._count + val

    def getCount(self):
        return self._count

# Straightforward implementation of the Singleton Pattern
# For storing lines in which comments needs to be added
class CommentsDict(object):
    _instance = None
    _commentsList = []
    def __new__(cls):
        if cls._instance is None:
            print('Creating the Comments object')
            cls._instance = super(CommentsDict, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance

    def addComment(self, lineNo, comment):
        self._commentsList.insert(0,tuple([lineNo, comment]))

    def getCommentList(self):
        return self._commentsList

    def hasComments(self):
        if len(self._commentsList):
            return True
        return False