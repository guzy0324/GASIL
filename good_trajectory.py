class GoodTrajectory:
    def __init__(self, T, R):
        self.T = T
        self.R = R
    def __lt__(self, other):
        return self.R < other.R
