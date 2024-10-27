import numpy as np
from scipy import stats

np.random.seed(3)
'''
    环境：定义作业，虚拟机，功能（读取作业，获取状态，反馈reward，记录各类数据）
'''

class SchedulingEnv:
    #   虚拟机和作业各类参数定义
    def __init__(self, args):
        #Environment Settings
        self.policy_num = len(args.Baselines)
        # 虚拟机
        self.VMtypes = args.VM_Type
        self.VMnum = args.VM_Num
        assert self.VMnum == len(self.VMtypes)
        self.VMcapacity = args.VM_capacity
        self.actionNum = args.VM_Num
        self.s_features = 1 + args.VM_Num  # 用于DQN输入
        self.VMAcc = args.VM_Acc
        self.VMCost = args.VM_Cost

        # 作业
        self.jobMI = args.Job_len_Mean
        self.jobMI_std = args.Job_len_Std
        self.job_type = args.Job_Type
        self.jobNum = args.Job_Num
        self.lamda = args.lamda
        self.arrival_Times = np.zeros(self.jobNum)
        self.jobsMI = np.zeros(self.jobNum)
        self.lengths = np.zeros(self.jobNum)
        self.types = np.zeros(self.jobNum)
        self.ddl = np.ones(self.jobNum) * args.Job_ddl
        # generate workload
        self.gen_workload(self.lamda)

        # 虚拟机和作业的状态记录
        # 1-VM id  2-start time  3-wait time  4-waitT+exeT  5-leave time  6-reward  7-actual_exeT  8- success  9-reject
        # 1-执行任务个数  2-执行结束时间
        # Random
        self.RAN_events = np.zeros((9, self.jobNum))
        self.RAN_VM_events = np.zeros((2, self.VMnum))
        # Round Robin
        self.RR_events = np.zeros((9, self.jobNum))
        self.RR_VM_events = np.zeros((2, self.VMnum))
        # Earliest
        self.early_events = np.zeros((9, self.jobNum))
        self.early_VM_events = np.zeros((2, self.VMnum))
        # DQN
        self.DQN_events = np.zeros((9, self.jobNum))
        self.DQN_VM_events = np.zeros((2, self.VMnum)) # end time of running VM

    #   生成工作负载（每个作业的计算量，到达时间，类型）
    def gen_workload(self, lamda):
        # 泊松分布生成时间间隔
        intervalT = stats.expon.rvs(scale=1 / lamda, size=self.jobNum)
        print("intervalT mean: ", round(np.mean(intervalT), 3),
              '  intervalT SD:', round(np.std(intervalT, ddof=1), 3))
        # 累加间隔时间生成每个工作的到达时间
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        last_arrivalT = self.arrival_Times[- 1]
        print('last job arrivalT:', round(last_arrivalT, 3))

        # 正态分布生成任务长度
        self.jobsMI = np.random.normal(self.jobMI, self.jobMI_std, self.jobNum)
        self.jobsMI = self.jobsMI.astype(int)
        print("MI mean: ", round(np.mean(self.jobsMI), 3), '  MI SD:', round(np.std(self.jobsMI, ddof=1), 3))
        self.lengths = self.jobsMI / self.VMcapacity
        print("length mean: ", round(np.mean(self.lengths), 3), '  length SD:', round(np.std(self.lengths, ddof=1), 3))

        # 生成任务类型
        types = np.zeros(self.jobNum)
        for i in range(self.jobNum):
            if np.random.uniform() < self.job_type:
                types[i] = 0
            else:
                types[i] = 1
        self.types = types

    #   重置环境
    def reset(self, args, jobNum_new, lamda_new, job_type_new  ):
        self.job_type = job_type_new #NEW
        self.jobNum = jobNum_new # NEW
   
        # if each episode generates new workload
        self.arrival_Times = np.zeros(self.jobNum)
        self.jobsMI = np.zeros(self.jobNum)
        self.lengths = np.zeros(self.jobNum)
        self.types = np.zeros(self.jobNum)
        self.ddl = np.ones(self.jobNum) * args.Job_ddl  # 250ms = waitT + exeT
        self.job_type = args.Job_Type
        self.gen_workload(lamda_new) #NEW
        self.VMtypes = args.VM_Type

        # reset all records
        self.SAIRL_events = np.zeros((9, self.jobNum))
        self.SAIRL_VM_events = np.zeros((2, self.VMnum))
        self.RAN_events = np.zeros((9, self.jobNum))
        self.RAN_VM_events = np.zeros((2, self.VMnum))
        self.RR_events = np.zeros((9, self.jobNum))
        self.RR_VM_events = np.zeros((2, self.VMnum))
        self.early_events = np.zeros((9, self.jobNum))
        self.early_VM_events = np.zeros((2, self.VMnum))
        self.DQN_events = np.zeros((9, self.jobNum))
        self.DQN_VM_events = np.zeros((2, self.VMnum))
        self.suit_events = np.zeros((9, self.jobNum))
        self.suit_VM_events = np.zeros((2, self.VMnum))
        self.sensible_events = np.zeros((9, self.jobNum))
        self.sensible_VM_events = np.zeros((2, self.VMnum))

    #   读取新的任务
    def workload(self, job_count):
        arrival_time = self.arrival_Times[job_count - 1]
        length = self.lengths[job_count - 1]
        jobType = self.types[job_count - 1]
        ddl = self.ddl[job_count - 1]
        if job_count == self.jobNum:
            finish = True
        else:
            finish = False
        job_attributes = [job_count - 1, arrival_time, length, jobType, ddl]
        return finish, job_attributes

    #   任务执行结束后，计算反馈reward并记录虚拟机和任务记录
    def feedback(self, job_attrs, action, policyID):
        job_id = job_attrs[0]
        arrival_time = job_attrs[1]
        length = job_attrs[2]
        job_type = job_attrs[3]
        ddl = job_attrs[4]
        acc = self.VMAcc[action]
        cost = self.VMCost[action]
        if job_type == self.VMtypes[action]:
            real_length = length / acc
        else:
            real_length = (length * 2) / acc

        if policyID == 1:
            idleT = self.RAN_VM_events[0, action]
        elif policyID == 2:
            idleT = self.RR_VM_events[0, action]
        elif policyID == 3:
            idleT = self.early_VM_events[0, action]
        elif policyID == 4:
            idleT = self.DQN_VM_events[0, action]

        # waitT & start exeT
        if idleT <= arrival_time:  # if no waitT
            waitT = 0
            startT = arrival_time
        else:
            waitT = idleT - arrival_time
            startT = idleT

        durationT = waitT + real_length  # waitT+exeT
        leaveT = startT + real_length  # leave T
        new_idleT = leaveT  # update VM idle time
        # reward
        #reward = - durationT / length
        #print('rrrrrrrrrrrrr', real_length)
        #print('lllllllllllll', length)
        #print('ddddddddddddd', durationT)
        #print('ccccccccccccc', cost)
        cost = 0.1 + real_length * cost
        reward = (1 + np.exp(1.5 - cost)) * (length) / durationT
        #reward = (length) / (durationT * cost)
        # whether success
        success = 1 if durationT <= ddl else 0

        if policyID == 1:   # random
            self.RAN_events[0, job_id] = action
            self.RAN_events[2, job_id] = waitT
            self.RAN_events[1, job_id] = startT
            self.RAN_events[3, job_id] = durationT
            self.RAN_events[4, job_id] = leaveT
            self.RAN_events[5, job_id] = reward
            self.RAN_events[6, job_id] = real_length
            self.RAN_events[7, job_id] = success
            self.RAN_events[8, job_id] = cost + np.random.rand() / 100
            # update VM info
            self.RAN_VM_events[1, action] += 1
            self.RAN_VM_events[0, action] = new_idleT
            # print('VMC_after:', self.RAN_VM_events[0, action])
        elif policyID == 2: # Round-Robin
            self.RR_events[0, job_id] = action
            self.RR_events[2, job_id] = waitT
            self.RR_events[1, job_id] = startT
            self.RR_events[3, job_id] = durationT
            self.RR_events[4, job_id] = leaveT
            self.RR_events[5, job_id] = reward
            self.RR_events[6, job_id] = real_length
            self.RR_events[7, job_id] = success
            self.RR_events[8, job_id] = cost + np.random.rand() / 100
            # update VM info
            self.RR_VM_events[1, action] += 1
            self.RR_VM_events[0, action] = new_idleT
        elif policyID == 3: # Earliest
            self.early_events[0, job_id] = action
            self.early_events[2, job_id] = waitT
            self.early_events[1, job_id] = startT
            self.early_events[3, job_id] = durationT
            self.early_events[4, job_id] = leaveT
            self.early_events[5, job_id] = reward
            self.early_events[6, job_id] = real_length
            self.early_events[7, job_id] = success
            self.early_events[8, job_id] = cost + np.random.rand() / 100
            # update VM info
            self.early_VM_events[1, action] += 1
            self.early_VM_events[0, action] = new_idleT
        elif policyID == 4: #DQN
            self.DQN_events[0, job_id] = action
            self.DQN_events[2, job_id] = waitT
            self.DQN_events[1, job_id] = startT
            self.DQN_events[3, job_id] = durationT
            self.DQN_events[4, job_id] = leaveT
            self.DQN_events[5, job_id] = reward
            self.DQN_events[6, job_id] = real_length
            self.DQN_events[7, job_id] = success
            self.DQN_events[8, job_id] = cost + np.random.rand() / 100
            # update VM info
            self.DQN_VM_events[1, action] += 1
            self.DQN_VM_events[0, action] = new_idleT
        return reward

    #   获取虚拟机执行结束时间
    def get_VM_idleT(self, policyID):
        if policyID == 3:
            idleTimes = self.early_VM_events[0, :]
        elif policyID == 4:
            idleTimes = self.DQN_VM_events[0, :]
        elif policyID == 5:
            idleTimes = self.suit_VM_events[0, :]
        elif policyID == 1:
            idleTimes = self.RAN_VM_events[0, :]
        elif policyID == 7:
            idleTimes = self.SAIRL_VM_events[0, :]
        elif policyID == 2:
            idleTimes = self.RR_VM_events[0, :]
        return idleTimes

    #   获取虚拟机和工作状态
    def getState(self, job_attrs, policyID):
        arrivalT = job_attrs[1]
        length = job_attrs[2]
        job_type = job_attrs[3]
        state_job = [job_type]
        if policyID == 4:  # DQN
            idleTimes = self.get_VM_idleT(4)
        elif policyID == 1:  # random
            idleTimes = self.get_VM_idleT(1)
        elif policyID == 2:  # RR
            idleTimes = self.get_VM_idleT(2)
        elif policyID == 3:  # used for as
            idleTimes = self.get_VM_idleT(3)
        elif policyID == 5:  # suitable
            idleTimes = self.get_VM_idleT(5)
        elif policyID == 7:  # SAIRL
            idleTimes = self.get_VM_idleT(7)
        waitTimes = [t - arrivalT for t in idleTimes]
        waitTimes = np.maximum(waitTimes, 0) # 去负值
        state = np.hstack((state_job, waitTimes))# [job_type,Vm_waitime]
        return state

    # 以下 计算各类指标用于实验评估
    def get_accumulateRewards(self, policies, start, end):

        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[5, start:end])
        rewards[1] = sum(self.RR_events[5, start:end])
        rewards[2] = sum(self.early_events[5, start:end])
        rewards[3] = sum(self.DQN_events[5, start:end])
        return np.around(rewards, 2)

    def get_accumulateCost(self, policies, start, end):

        Cost = np.zeros(policies)
        Cost[0] = sum(self.RAN_events[8, start:end])
        Cost[1] = sum(self.RR_events[8, start:end])
        Cost[2] = sum(self.early_events[8, start:end])
        Cost[3] = sum(self.DQN_events[8, start:end])
        return np.around(Cost, 2)

    def get_FinishTimes(self, policies, start, end):
        finishT = np.zeros(policies)
        finishT[0] = max(self.RAN_events[4, start:end])
        finishT[1] = max(self.RR_events[4, start:end])
        finishT[2] = max(self.early_events[4, start:end])
        finishT[3] = max(self.DQN_events[4, start:end])
        return np.around(finishT, 2)

    def get_executeTs(self, policies, start, end):
        executeTs = np.zeros(policies)
        executeTs[0] = np.mean(self.RAN_events[6, start:end])
        executeTs[1] = np.mean(self.RR_events[6, start:end])
        executeTs[2] = np.mean(self.early_events[6, start:end])
        executeTs[3] = np.mean(self.DQN_events[6, start:end])
        return np.around(executeTs, 3)

    def get_waitTs(self, policies, start, end):
        waitTs = np.zeros(policies)
        waitTs[0] = np.mean(self.RAN_events[2, start:end])
        waitTs[1] = np.mean(self.RR_events[2, start:end])
        waitTs[2] = np.mean(self.early_events[2, start:end])
        waitTs[3] = np.mean(self.DQN_events[2, start:end])
        return np.around(waitTs, 3)

    def get_responseTs(self, policies, start, end):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[3, start:end])
        respTs[1] = np.mean(self.RR_events[3, start:end])
        respTs[2] = np.mean(self.early_events[3, start:end])
        respTs[3] = np.mean(self.DQN_events[3, start:end])
        return np.around(respTs, 3)

    def get_successTimes(self, policies, start, end):
        successT = np.zeros(policies)
        successT[0] = sum(self.RAN_events[7, start:end]) / (end - start)
        successT[1] = sum(self.RR_events[7, start:end]) / (end - start)
        successT[2] = sum(self.early_events[7, start:end]) / (end - start)
        successT[3] = sum(self.DQN_events[7, start:end]) / (end - start)
        successT = np.around(successT, 3)
        return successT

    def get_rejectTimes(self, policies, start, end):
        reject = np.zeros(policies)
        reject[0] = sum(self.RAN_events[8, start:end])
        reject[1] = sum(self.RR_events[8, start:end])
        reject[2] = sum(self.early_events[8, start:end])
        reject[3] = sum(self.DQN_events[8, start:end])
        return np.around(reject, 2)

    def get_totalRewards(self, policies, start):
        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[5, start:self.jobNum])
        rewards[1] = sum(self.RR_events[5, start:self.jobNum])
        rewards[2] = sum(self.early_events[5, start:self.jobNum])
        rewards[3] = sum(self.DQN_events[5, start:self.jobNum])
        return np.around(rewards, 2)

    def get_totalTimes(self, policies, start):
        finishT = np.zeros(policies)
        finishT[0] = max(self.RAN_events[4, :]) - self.arrival_Times[start]
        finishT[1] = max(self.RR_events[4, :]) - self.arrival_Times[start]
        finishT[2] = max(self.early_events[4, :]) - self.arrival_Times[start]
        finishT[3] = max(self.DQN_events[4, :]) - self.arrival_Times[start]
        return np.around(finishT, 2)

    def get_avgUtilitizationRate(self, policies, start):
        avgRate = np.zeros(policies)  # sum(real_length)/ totalT*VMnum
        avgRate[0] = sum(self.RAN_events[6, start:self.jobNum]) / (
                    (max(self.RAN_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        avgRate[1] = sum(self.RR_events[6, start:self.jobNum]) / (
                    (max(self.RR_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        avgRate[2] = sum(self.early_events[6, start:self.jobNum]) / (
                    (max(self.early_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        avgRate[3] = sum(self.DQN_events[6, start:self.jobNum]) / (
                    (max(self.DQN_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        return np.around(avgRate, 3)

    def get_all_responseTs(self, policies):
        respTs = np.zeros((policies, self.jobNum))
        respTs[0, :] = self.RAN_events[3, :]
        respTs[1, :] = self.RR_events[3, :]
        respTs[2, :] = self.early_events[3, :]
        respTs[3, :] = self.DQN_events[3, :]
        return np.around(respTs, 3)

    def get_total_responseTs(self, policies, start):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[3, start:self.jobNum])
        respTs[1] = np.mean(self.RR_events[3, start:self.jobNum])
        respTs[2] = np.mean(self.early_events[3, start:self.jobNum])
        respTs[3] = np.mean(self.DQN_events[3, start:self.jobNum])
        return np.around(respTs, 3)

    def get_totalSuccess(self, policies, start):
        successT = np.zeros(policies)  # sum(self.RAN_events[7, 3000:-1])/(self.jobNum - 3000)
        successT[0] = sum(self.RAN_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        successT[1] = sum(self.RR_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        successT[2] = sum(self.early_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        successT[3] = sum(self.DQN_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        print(self.jobNum)
        return np.around(successT, 3)

    def get_totalCost(self, policies, start):

        Cost = np.zeros(policies)
        Cost[0] = sum(self.RAN_events[8, start:self.jobNum]) / (self.jobNum - start + 1)
        Cost[1] = sum(self.RR_events[8, start:self.jobNum]) / (self.jobNum - start + 1)
        Cost[2] = sum(self.early_events[8, start:self.jobNum]) / (self.jobNum - start + 1)
        Cost[3] = sum(self.DQN_events[8, start:self.jobNum]) / (self.jobNum - start + 1)
        return np.around(Cost, 3)
    
    def printttt(self):
        nums = self.policy_num
        print(nums)