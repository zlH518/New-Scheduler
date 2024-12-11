from waitingList import WaitingList
from migrateSolver import MigrateSolver

class Scheduler:
    def __init__(self,schedulerConfig):
        self.schedulerConfig = schedulerConfig
        
        if schedulerConfig['waitingList'] is not None:
            self.waitingListConfig = schedulerConfig['waitingList']
            self.waintingList = WaitingList(self.waitingListConfig)

        if schedulerConfig['migrateSolver'] is not None:
            self.migrateSolverConfig = schedulerConfig['migrateSolver']
            self.migrateSolver = MigrateSolver(self.migrateSolverConfig)




        
        
