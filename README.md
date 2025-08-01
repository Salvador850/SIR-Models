Progams

Agent Based Modeling

Program Summary: 
This Program is an Agent Based model of epidemics which uses perlin noise generation to hotspots, creating ciy like clusters of agents each agent has three main modes of movement, one where it moves randomly, one where it is attracted to hotspots and 
one where it travels to a predesignated home to simulate a lockdown. The frequency of each of these movement modes can be customized. In addition this model also allows for the simulation of vital dynamics(births and deaths), sleep cycles, vaccine 
development, and lockdowns.

Command Line Arguments: 
The command arguments are as follows: clumping_factor changes how attracted agents are to hotspots on the map this value should be a float above zero and below one. The next value map_size should be a positive integer, the map with be a square with width
map_size. The third argument is infection_time which represents the average amount of time that the agents are infected for. The next argument is recovery_time it should be a positive integer that represents the average amount of days and agent is immune to the disease for. The fifth 
argument is agent_count and should be a positive integer representing the amount of agents in the simulation. The next value is infection_rate, it is a positive float, this value is a y axis dilation that is applied to the function f(x) = 2^x where x is 
the distance from an infected agent and a susceptible one, and the output is the probability of infection at that time step. the next argument is vaccine_date, it should be a positive integer and this controls the timestep where some agents will have a
simulated vaccine developed with a 90% efficacy. The next three arguments control vital dynamics and should all be positive floats. These arguments are birth_rate(controls birth rate), death_rate, which control the natural death rate, and fatality_rate,
which controls the fatality rate of the virus. The eleventh argument is sleep_cycle_on, this value should be a boolean, and it controls whether or not a simulated sleep cycle, occuring every 150 steps is activated
The next argument is globalization, which cntrols the rate of regional and global travel in the simulation, and should be a float equal to above zero and below one. The following argument is fear which applies a y axis dilation to equation 
fear = infected/total. The value should be a positive float. The final argument is max_vacc, which controls the maximum percentage of the population which will become vaccinated, this should be a float between zero and one.

Dependencies:
numpy, matplotlib, celluloid, progress, noise 

Pasteable Command to Get You Started:
python3 agent_based_SIR.py 0.95 500 30 100 500 2000 0.8 1500 0.001 0.001 0.075 False 0.5 0.0


SIRS

Program Summary:
Simple deterministic model used to predict diseases, there is an option to compare the data to other models or real world data using mean squared error

Command Line arguments:

1. initial amount of susceptible individuals(integer)
2. initial amount of infected individuals(integer)
3. initial amount of recovered indivuals(integer)
4. infection rate(float)
5. recovery rate(float)
6. suceptibility rate(float)
7. birth rate(float)
8. death rate(float)
9. division factor(divide outputs by x amount)(float)
10 and above: files of data to compare against SIRS model, should be a list of case numbers over time, the file format should be .txt

Dependencies:
scipy, pandas, numpy, matplotlib, progress

Pasteable Command:
ython3 SIRS.py 1500 5 0 0.16 0.07 0.009 0.001 0.001 30


Chronos Forecasting Model

Program Summary:
Trains Chronos bolt_base Models on given data, training on progressively more time series, or generates zero shot model to predict provided data, outputs Chronos.

Command Line Arguments:

1. Data File, should be a 2d list, each sublist should have three items: the time series id, timestamp, and case numbers. The file format should be .txt
2. This argument should be a boolean, True indicates that you want fine tuned models, false indicates that you would like a zero shot model.
3. The maximum amount of time series the Chronos models should be trained on(integer).

Dependencies:
pandas, torch, progress, autogluon

Pasteable Command:
python3 chronos_forecasting_model.py "/home/some data.txt" True

Chronos Real World Predict

Program Summary:
Test a zero shot and fine tuned model on a large time series

Command Line Arguments:
1. Data File, should be a list of case numbers over time, .txt file format.
2. Fine Tuned Model Path, should be the path to a folder that contains all data for a chronos model(this should be pregenerated when you create a chronos model)
3. Prediction Lenth, this is the prediction length of the fine tuned model that your file path leads to(integer).

Dependencies
autogluon, progress, torch, matplotlib, pandas



