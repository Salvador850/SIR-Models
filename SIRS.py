import sys
import pandas as pd
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar


data_frame = None
for i in range(10, len(sys.argv)):
        data_file = open(sys.argv[i], 'r').read()
        target_list = eval(data_file)[:1000]
        timestamp_list = [i for i in range(len(target_list))]
        item_id_list = [0 for i in range(len(target_list))]

        full_data = zip(item_id_list, timestamp_list, target_list)
        additional_data = pd.DataFrame(full_data, columns=["item_id", "timestamp", "target"])

        additional_data['target'] /= 4 * (10**6)
        if data_frame != None:
            data_frame = pd.concat([data_frame, additional_data], ignore_index=True)
        else:
            data_frame = additional_data
# data_lists = [i / 30 for i in data_lists]

# SIR model equations
def SIRS_model(y, t, beta, gamma, theta, b_rate, d_rate):
    S, I, R = y
    total = S + I + R
    dSdt = (b_rate * total) - ((beta * S * I) / (S + I + R)) + (theta * R) - (d_rate * S) 
    dIdt = ((beta * S * I) / (S + I + R))  - (gamma * I) - (d_rate * I)
    dRdt = gamma * I - (theta * R) - (d_rate * R)
    return [dSdt, dIdt, dRdt]


y0 = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]

beta = float(sys.argv[4])
gamma = float(sys.argv[5])
theta = float(sys.argv[6])

b_rate = float(sys.argv[7])
d_rate = float(sys.argv[8])

time_series_len = len(data_frame)
t = np.linspace(0, time_series_len, time_series_len)
mse = []

solution = odeint(SIRS_model, y0, t, args=(beta, gamma, theta, b_rate, d_rate))
S, I, R = solution.T
new_I = [0] + [(beta * s * i) / (s + i + r) for s, i, r in zip(S, I, R)]
new_I = [i / int(sys.argv[9]) for i in new_I]
data_lists = list(data_frame['target'])

for data_point in range(time_series_len):
    mse.append((data_lists[data_point] - new_I[data_point])**2)
    

plt.plot([data_point for data_point in data_lists], label="Agent Based Output")
plt.plot(new_I, label="SIRS Output", color="orange")
plt.xlabel('Timesteps')
plt.ylabel('New Case Predictions')
plt.title('SIRS vs. Agent Based Model Results Medium Urbanization')

plt.legend()
plt.show()

frame_mse = []           
for i in range(19):
   frame = mse[i*50:][:(i + 1) * 50]
   frame_mse.append(sum(frame)/len(frame))

print("Frame Error:", frame_mse)
print("Error:", sum(mse)/len(mse))
