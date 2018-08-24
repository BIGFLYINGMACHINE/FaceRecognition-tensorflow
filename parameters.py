import csv
BATCH_SIZE = 128
IMG_SIZE = 64
NUM_CHANNELS = 3
# 第一层卷积层的尺寸与深度
CONV1_SIZE = 3
CONV1_DEPTH = 18
# 第二层
CONV2_SIZE = 3
CONV2_DEPTH = 32
# 第三层
CONV3_SIZE = 3
CONV3_DEPTH = 45
# 第四层
CONV4_SIZE = 3
CONV4_DEPTH = 40
# 全连接层1
FC1_SIZE = 108
FC2_SIZE = 64
OUTPUT_SIZE = 2


LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.97
DECAY_STEPS = 100 # 与学习率的指数衰减有关，此数值越大衰减越慢
# decayed_learning_rate = learning_rate_base *
#                        decay_rate ^ (global_step / decay_steps)
REGULARAZATION_RATE = 0.0005
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"


# 定义了保存参数以及运行结果的函数
def save_parameters_as_csv(outcome, csv_path):
    paras = {"LEARNING_RATE_BASE" : LEARNING_RATE_BASE, 
            "LEARNING_RATE_DECAY" : LEARNING_RATE_DECAY,
            "REGULARAZATION_RATE" : REGULARAZATION_RATE,
            "TRAINING_STEPS" : TRAINING_STEPS,
            "MOVING_AVERAGE_DECAY" : MOVING_AVERAGE_DECAY,
            "DECAY_STEPS" : DECAY_STEPS}
    paras = dict(paras, **outcome) # 合并两个字典
    para_name = []
    para_value = []
    for key, value in paras.items():    
        para_name.append(key)
        para_value.append(value)
    csvFile = open(csv_path, "a+") # 设置为添加模式
    writer = csv.writer(csvFile)
    writer.writerow(para_name)
    writer.writerow(para_value)
    csvFile.close()