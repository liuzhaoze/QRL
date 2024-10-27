import argparse


# argparse 命令行选项、参数和子命令解析器

"""
    定义虚拟机，任务，DQN，训练过程的各项参数
    集中定义，方便调整
"""

def parameter_parser():
    """
    ArgumentParser对象:
        description - 在参数帮助文档之后显示的文本
    """
    parser = argparse.ArgumentParser(description="SAIRL")


    # General
    """
    add_argument()方法

        name or flags - 一个命名或者一个选项字符串的列表
        
        action - 表示该选项要执行的操作
        
        default - 当参数未在命令行中出现时使用的值
        
        dest - 用来指定参数的位置
        
        type - 为参数类型，例如int
        
        choices - 用来选择输入参数的范围。例如choice = [1, 5, 10], 表示输入参数只能为1,5 或10
        
        help - 用来描述这个选项的作用
    """
    #对比方法参数
    parser.add_argument("--Baselines",
                        type=list,
                        default=['Random', 'Round-Robin', 'Earliest', 'DQN'],
                        help="Experiment Baseline")
    parser.add_argument("--Baseline_num",
                        type=int,
                        default=4,
                        help="Number of baselines")

    # 训练参数
    #   训练周期
    parser.add_argument("--Epoch",
                        type=int,
                        default=5,
                        help="Training Epochs")

    # DQN
    #   开始训练时间
    parser.add_argument("--Dqn_start_learn",
                        type=int,
                        default=500,
                        help="Iteration start Learn for normal dqn")
    #   学习频率
    parser.add_argument("--Dqn_learn_interval",
                        type=int,
                        default=1,
                        help="Dqn's learning interval")
    #   学习率
    parser.add_argument("--Lr_DDQN",
                        type=float,
                        default=0.001,
                        help="Dueling DQN Lr")

    # 虚拟机参数
    #   类型：High I\O or High CPU
    parser.add_argument("--VM_Type",
                        type=list,
                        default=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        help="VM Type")
    #   成本
    parser.add_argument("--VM_Cost",
                        type=list,
                        default=[1, 1, 2, 2, 4, 1, 1, 2, 2, 4],
                        help="VM Cost")
    #   VCPU数量
    parser.add_argument("--VM_Acc",
                        type=list,
                        default=[1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2],
                        help="VM Cpus")
    #   虚拟机数量
    parser.add_argument("--VM_Num",
                        type=int,
                        default=10,
                        help="The number of VMs")
    #   虚拟机计算能力
    parser.add_argument("--VM_capacity",
                        type=int,
                        default=1000,
                        help="VM capacity")

    # 任务参数
    #   平均到达速度
    parser.add_argument("--lamda",
                        type=int,
                        default=20,
                        help="The parameter used to control the length of each jobs.")
    #   类型比例 IO sensitive or Computing sensitive
    parser.add_argument("--Job_Type",
                        type=float,
                        default=0.9,
                        help="The parameter used to control the type of each jobs.")
    #   数量
    parser.add_argument("--Job_Num",
                        type=int,
                        default=8000,
                        help="The number of jobs.")
    #   平均计算量
    parser.add_argument("--Job_len_Mean",
                        type=int,
                        default=200,
                        help="The mean value of the normal distribution.")
    #   计算量标准差
    parser.add_argument("--Job_len_Std",
                        type=int,
                        default=20,
                        help="The std value of the normal distribution.")
    #   QoS 响应时间要求
    parser.add_argument("--Job_ddl",
                        type=float,
                        default=0.25,
                        help="Deadline time of each jobs")
    # Plot
    # parser.add_argument("--Plot_labels",
    #                     type=list,
    #                     default=['b-', 'm-', 'g-', 'y-', 'r-', 'k-', 'w-'],
    #                     help="Deadline time of each jobs")
    return parser.parse_args()
