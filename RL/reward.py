
class Reward:
    def __init__(self, omega):
        """
        omega: list[2], sum@omega == 1.0
        """
        self.omega1 = omega[0]
        self.omega2 = omega[1]
    
    def getReward(self, thrput, latency):
        return self.omega1 * thrput + self.omega2 * latency