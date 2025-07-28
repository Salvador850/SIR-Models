import sys
import pandas as pd
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

for i in range(11, len(sys.argv)):
    data_file = open(sys.argv[i], 'r').read()
    if ']]' in data_file:
        data_lists = data_file.split(']]')
        
        data_lists =  '[' + ']], '.join(data_lists)[:-2].replace('None', '') + ']'
        data_lists = eval(data_lists)
    else:
        data_file = open(sys.argv[i], 'r').read()
        target_list = eval(data_file)
        timestamp_list = [i for i in range(len(target_list))]
        item_id_list = [0 for i in range(len(target_list))]

        full_data = zip(item_id_list, timestamp_list, target_list)
        data_frame = pd.DataFrame(full_data, columns=["item_id", "timestamp", "target"])

        data_frame['target'] /= 2 * (10**6)
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

time_series_len = 500
t = np.linspace(0, time_series_len, time_series_len)
mse = []

bar = Bar('Running Simulation', max=len(data_lists))

solution = odeint(SIRS_model, y0, t, args=(beta, gamma, theta, b_rate, d_rate))
S, I, R = solution.T
new_I = [0] + [(beta * s * i) / (s + i + r) for s, i, r in zip(S, I, R)]
new_I = [i / sys.argv[10] for i in new_I]


for i in range(len(data_lists)):
    for data_point in range(time_series_len):
        mse.append((data_lists[i][data_point][-1] - new_I[data_point])**2)
        
    if i % 9 == 0:
        plt.plot(t,  [data_point[-1] for data_point in data_lists[i]], label="Agent Based Output")
        plt.plot(new_I, label="SIRS Output", color="orange")
        """
        plt.plot(S, label='Susceptible', color='blue')
        plt.plot(I, label='Infected',  color='red')
        plt.plot(R, label='Recovered', color='green')
        plt.plot([S[i] + I[i] + R[i] for i in range(len(S))], label='Total', color='black')
        """
        plt.xlabel('Timesteps')
        plt.ylabel('New Case Predictions')
        plt.title('SIRS vs. Agent Based Model Results Medium Urbanization')
        
        plt.legend()
        plt.show()
    bar.next()
            
bar.finish()


print("Error:", sum(mse)/len(mse))
