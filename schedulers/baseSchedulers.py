from waitingList import WaitingList
from migrateSolver import MigrateSolver

class Scheduler:
    def __init__(self,schedulerConfig):
        self.schedulerConfig = schedulerConfig
        self.waitingListConfig = schedulerConfig['waitingList']
        self.migrateSolverConfig = schedulerConfig['migrateSolver']

        self.waintingList = WaitingList(self.waitingListConfig)
        self.migrateSolver = MigrateSolver(self.migrateSolverConfig)




        
        
