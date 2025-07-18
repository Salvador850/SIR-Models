Program Summary: 
This Program is an Agent Based model of epidemics which uses perlin noise generation to hotspots, creating ciy like clusters of agents each agent has three main modes of movement, one where it moves randomly, one where it is attracted to hotspots and 
one where it travels to a predesignated home to simulate a lockdown. The frequency of each of these movement modes can be customized. In addition this model also allows for the simulation of vital dynamics(births and deaths), sleep cycles, vaccine 
development, and lockdowns.

Command Line Arguments: 
The command arguments are as follows: clumping_factor changes how attracted agents are to hotspots on the map this value should be a float above zero and below one. The next value map_size should be a positive integer, the map with be a square with width
map_size. The third argument is infection_time which represents the average amount of time that the agents are infected for. The next argument is recovery_time it should be a positive integer that represents the average amount of days. The fifth 
argument is agent_count and should be a positive integer representing the amount of agents in the simulation. The next value is infection_rate, it is a positive float, this value is a y axis dilation that is applied to the function f(x) = 2^x where x is 
the distance from an infected agent and a susceptible one, and the output is the probability of infection at that time step. the next argument is vaccine_date, it should be a positive integer and this controls the timestep where some agents will have a
simulated vaccine developed with a 90% efficacy. The next three arguments control vital dynamics and should all be positive floats. These arguments are birth_rate(controls birth rate), death_rate, which control the natural death rate, and fatality_rate,
which controls the fatality rate of the virus. The eleventh argument is sleep_cycle_on, this value should be a boolean, and it controls whether or not a simulated sleep cycle, occuring every 150 steps is activated
The next argument is globalization, which cntrols the rate of regional and global travel in the simulation, and should be a float equal to above zero and below one. The following argument is fear which applies a y axis dilation to equation 
fear = infected/total. The value should be a positive float. The final argument is max_vacc, which controls the maximum percentage of the population which will become vaccinated, this should be a float between zero and one.
