import numpy as np

demonstrations = np.load("res_demos.npz")["demos"]

for example in range(demonstrations.shape[0]):
    img_batch = demonstrations[example, 0]["arr_0"]
    dems = demonstrations[example,1:]
    label_dict = {0:(0.0, 0.0, 0.0), 1:(0.0, 0.0, 0.0), 2:(0.0, 0.0, 0.0), 3:(0.0, 0.0, 0.0), 4:(0.0, 0.0, 0.0)}
    for col in range(len(dems)):
        if col < len(dems) - 1:
            action = dems[col]
            reward = dem[col + 1]
            prev_total, prev_num, prev_avg = label_dict[action]
            label_dict[action] = (prev_total + reward, prev_num + 1, prev_total/prev_num)
    label = np.zeros(5)
    for i in range(len(label)):
        label[i] = label_dict[i][2]


    
