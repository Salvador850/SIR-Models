import random
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from noise import pnoise2
from matplotlib import cm
from celluloid import Camera
from progress.bar import Bar

class Agent:
    def __init__(self, hotspot_map, map_size, family_id, intelligence, infection_status=0):
        self.map_size = [map_size]*2
        self.coords = (random.randint(0, self.map_size[0]), random.randint(0, self.map_size[1]))
        self.hotspot_map = hotspot_map
        self.infection_status = infection_status
        self.trajectory = []
        self.new_trajectory()
        self.family_id = family_id
        self.steps_left = 0
        self.home_coords = self.coords
        self.intelligence = intelligence
        self.is_new = False

    #moves along the path that ideal_trajectory has calculated to have more hotspots
    #there is some chance the agent moves along a random trajectory
    def move(self, clumping_factor, agents, fear, timestep=3):
        self.coords = [self.trajectory[0] + self.coords[0], self.trajectory[1] + self.coords[1]]
        trajectories = self.hotspot_trajectory()

        move_avoid = self.avoid_move(agents, fear, clumping_factor, timestep)
        if move_avoid == True:
            return
        
        move_rand = self.rand_move(clumping_factor)
        if move_rand == True:
            return
        
        if trajectories[0][1] > trajectories[0][0]:
            self.trajectory = trajectories[1][1]

    # agent moves to attempt to avoid the virus
    def avoid_move(self, agents, fear, clumping_factor, timestep):
        go_home_odds = (0.075 / 25) * (fear * 200)
       
        if random.random() < go_home_odds or self.steps_left > 100 or timestep % 150 == 0:
            self.steps_left -= 1
            self.go_home()
            if self.steps_left < 100:
                self.steps_left = 160

            if self.steps_left == 100:
                self.steps_left = 0
            return True
    
        if random.random() < 0.2 * fear or self.steps_left > 200:
            self.steps_left -= 1
            self.social_distance(agents)
            if self.steps_left < 200:
                self.steps_left = 210

            if self.steps_left == 200:
                self.steps_left = 0
            return True


        return False
        
    # the agent moves randomly, either over a short or long range
    def rand_move(self, clumping_factor):
        self.coords = [self.coords[0] + self.trajectory[0], self.coords[1] + self.trajectory[1]]
        globalization = 1
        if len(sys.argv) > 13:
            globalization = float(sys.argv[13])
            
        if self.steps_left > 0 or random.random() < (0.075 / 120) * globalization:
            if self.steps_left == 0:
                self.trajectory = [random.randint(-25, 25) , random.randint(-25, 25)]
                self.steps_left = random.randint(3, 7)                
            self.steps_left -= 1
            return True
            
        if random.random() < (1 - clumping_factor) * 0.25 or self.steps_left < 0:
            if self.steps_left == 0:
                self.new_trajectory()
                self.steps_left = -random.randint(10, 20)
            self.steps_left += 1  
            return True
                
        if random.random() < (0.075 / 80) * globalization or self.steps_left > 300:
            self.steps_left -= 1
            if self.steps_left < 300:
                self.trajectory = [random.randint(-12, 12) , random.randint(-12, 12)]
                self.steps_left = random.randint(303, 307)
            if self.steps_left == 300:
                self.steps_left = 0
            
            return True

        return False
        
    #calculates which of two random trajectories has more hotspots along it's path
    def hotspot_trajectory(self):
        self.coords = [self.coords[0] + self.trajectory[0], self.coords[1] + self.trajectory[1]]
        self.check_out_of_bounds()
        trajectories = [self.trajectory, [(random.random()*2) - 1, (random.random()*2) - 1]]
        weighted_avgs = []
        
        for i in range(2):
            hotspot_frequency = 0
            weights_sum = 0.1    
            future_steps = 3
            for j in range(50):
                future_coords = [ round( self.coords[0] + (trajectories[i][0] * future_steps) ), \
                                  round( self.coords[1] + (trajectories[i][1] * future_steps) ) ]
                
                if future_coords[0] >= self.map_size[0] or future_coords[0] <= 0 or future_coords[1] >= self.map_size[1] or future_coords[1] <= 0:
                    break
                
                distance = math.sqrt( ((future_coords[0] - self.coords[0])  ** 2 ) + ((future_coords[1] - self.coords[1]) ** 2) )
                hotspot_frequency += (2**(-distance*0.2) ) * self.hotspot_map[future_coords[0]][future_coords[1]]
                weights_sum += 2**(-distance*0.2)
                future_steps += 3

            weighted_avgs.append(hotspot_frequency / weights_sum)

        return [weighted_avgs, trajectories]
    

                 
    def check_out_of_bounds(self):
        #teleports agents across the map if they go out of bounds
        if self.coords[0] > self.map_size[0]:
            self.coords[0] -= self.map_size[0]
        if self.coords[1] > self.map_size[1]:
            self.coords[1] -= self.map_size[1]
        if self.coords[0] < 0:
            self.coords[0] += self.map_size[0]
        if self.coords[1] < 0:
            self.coords[1] += self.map_size[1]

    #set random trajectory
    def new_trajectory(self):
        self.trajectory = [(random.random() * 5) - 2.5, (random.random()*5) - 2.5 ]
       

        
    # infected and recovered agents move one step closer to recovery/suseptibility, agents who are
    # suseptible have an exponentially lower chance of infection from other distant agents                  
    def infection(self, agent_list, infection_time, recovery_time, transmissibility):
        if self.infection_status < 0:
            self.infection_status += 1
            return 
        elif self.infection_status == 1:
            self.infection_status = -random.randint(max(recovery_time - 5, 10), max(15, recovery_time + 5) )
            return
        elif self.infection_status > 0:
            self.is_new = False
            self.infection_status -= 1
            return 

        # If the agent isn't infected, it has a chance getting the virus from nearby agents
        for agent in agent_list:
            if agent[1] > 0:
               rand = random.random()
               distance = math.sqrt( ((agent[0][0] - self.coords[0])  ** 2 ) + ((agent[0][1] - self.coords[1]) ** 2) )
               
               infection_odds = 2**(-distance * (1/transmissibility)) 
               if rand < infection_odds:
                   self.infection_status = random.randint(infection_time - 5, infection_time + 5)
                   self.is_new = True
                   break

    # moves towards one of the family's spawn
    def go_home(self):
        self.coords = [self.coords[0] + self.trajectory[0], self.coords[1] + self.trajectory[1]]
        self.check_out_of_bounds()
        self.trajectory = [(self.home_coords[0] - self.coords[0]) / 50, (self.home_coords[1] - self.coords[1]) / 50]

    # finds one of the family's spawn
    def find_home(self, agents):
        home_coords = []
        for agent in agents:
            if agent.get_family_id() == self.family_id:
                home_coords.append(agent.get_coords())
 
        self.home_coords = [i + (random.random()*2) - 1 for i in home_coords[-1]]

    # moves away from nearby agents
    def social_distance(self, agents):
        min_distance = None
        min_distance_coord = None
        for agent in agents:
            distance = math.sqrt( ((agent.get_coords()[0] - self.coords[0])  ** 2 ) + ((agent.get_coords()[1] - self.coords[1]) ** 2) )
            if distance == 0:
                continue
            if distance < 15:
                min_distance_coord = agent.get_coords()
                self.avoid_infection(min_distance_coord)
                break
                
        self.coords = [self.coords[0] + self.trajectory[0], self.coords[1] + self.trajectory[1]]
        self.check_out_of_bounds()

    # moves away from given coords
    def avoid_infection(self, other_coord):
        trajectory = [other_coord[0] - self.coords[0], other_coord[1] - self.coords[1]]
        trajectory = [i/5 for i in trajectory]
        self.trajectory = [-i*random.randint(1, 5) for i in trajectory]
                        
    def infection_status(self):
        return self.infection_status

    def get_coords(self):
        return self.coords

    def get_id(self):
        return self.family_id

    def get_family_id(self):
        return self.family_id
    
    def set_family_id(self, value):
        self.family_id = value

    def set_fear(self, value):
        self.fear = value

    def get_intelligence(self):
        return self.intelligence


def recieve_inputs():
    terminal_inputs = [0.85, 150, 14, 25, 200, 100, 1, -1, 0.0001, 0.00008, 0.05, False]
    for arg in range(len(terminal_inputs)):
        if arg > len(sys.argv) - 2:
            break
        if not arg in [0, 6, 8, 9, 10, 11]:
            terminal_inputs[arg] = int(sys.argv[arg + 1])
        elif arg != 11:
            terminal_inputs[arg] = float(sys.argv[arg + 1])
        else:
            terminal_inputs[arg] = eval(sys.argv[arg + 1])
            
    if terminal_inputs[7] == -1:
        terminal_inputs[7] = round(terminal_inputs[4] * 0.8)
    """
    clumping_factor = sys.argv[0] map_size = sys.argv[1] infection_time = sys.argv[2]
    recovery_time = sys.argv[3] timesteps = sys.argv[4] agent_count = sys.argv[5] transmissibility = sys.argv[6]
    vaccine_date = sys.argv[7], birth_rate = sys.argv[8] death_rate = sys.argv[9] fatality_rate = sys.argv[10]
    sleep_cycle_on = sys.argv[11]
    """
    # for i in range(100):
        # terminal_inputs[6] += ((random.random() * 2) - 1) / 10
    print(simulation(terminal_inputs, 0))


    
#creates graph animation and runs simulation: blue = susceptible, red = infected, green = recovered
def simulation(params, step):
    global SIR_count
    global agents_list
    SIR_count = [[], [], [], [], []]
    data_table = []
    # print("generating hotspot maps...")
    hotspot_maps = [hotspot_generation(params[1], bell_curve(0.5, 0.2, 0.05, 0.95), bell_curve(6, 1, 2, 8)) for i in range(5)]
    agents_list = new_agents(hotspot_maps, params[5], params[2], params[1], True)
    virus_spreading = False
    fig = plt.figure()
    camera = Camera(fig)
    bar = Bar('Running Simulation', max=params[4])
    
   #  print("loading agents...")
    for i in range(params[4] + 75):
        if i == 75:
            # print("simulating virus spread...")            
            virus_spreading = True
        if i == 0:
            # print("generating agent IDs...")
            generate_families(agents_list, agents_list)

        if i == 25:
            for agent in agents_list:
                agent.find_home(agents_list)

        if len(agents_list) == 0:
            update_graphs(agents_list, i, params[4], camera)
            break
       
        colors = generate_scatter_colors(agents_list)
        fear = (colors.count([1, 0, 0]) / len(agents_list)) * 0.5
        if len(sys.argv) > 14:
            fear = (colors.count([1, 0, 0]) / len(agents_list)) * float(sys.argv[14])

        timestep(agents_list, virus_spreading, params, i, fear)
        if virus_spreading:
            bar.next()
            update_graphs(agents_list, i, params[4], camera)
            data_table.append([int(((i - 75) + (step * 500))/500), i, *[i[-1] for i in SIR_count]])
            vital_dynamics(hotspot_maps, params)

    bar.finish()
   # return data_table


#updates graph data for each timestep in the simulation
def update_graphs(agents, timestep, max_time, camera):
    global SIR_count
    global agents_list
    if timestep < max_time + 74 and len(agents) != 0:
        colors = generate_scatter_colors(agents_list)
        SIR_count[0].append(colors.count([0, 0, 1]))
        SIR_count[1].append(colors.count([1, 0, 0]))
        SIR_count[2].append(colors.count([0, 1, 0]))
        SIR_count[3].append(len(colors))
        SIR_count[4].append(len([agent for agent in agents if agent.is_new == True]))
        # terrain_plot = plt.imshow(hotspots, cmap='Greys', zorder=1)
        plt.scatter([agent.coords[0] for agent in agents], [agent.coords[1] for agent in agents], c=colors)
        camera.snap()
        return
   
    anim = camera.animate(interval=30, blit=True)
    plt.show()
    print(plot_results(SIR_count))

    
#plots the amount of agents in each SIR category for each timestep on line graph
def plot_results(results):
    plt.plot(results[0], label='Susceptible', color='blue')
    plt.plot(results[1], label='Infected',  color='red')
    plt.plot(results[2], label='Recovered', color='green')
    plt.plot(results[3], label='Total', color='black')
    plt.plot(results[4], label='New Cases', color='orange')

    plt.xlabel('Timesteps')
    plt.ylabel('Amount of Agents in Each SIR Category')
    plt.title('SIR Model Results')

    plt.legend()
    plt.show()


# generates a color for each agent based on whether it is S, I, or R
def generate_scatter_colors(agents):
    colors = []
    for agent in agents:
        if agent.infection_status == 0:
            colors.append([0, 0, 1])
        elif agent.infection_status > 0:
            colors.append([1, 0, 0])
        elif agent.infection_status < 0:
            colors.append([0, 1, 0])

    return colors


def vaccination(x, mean, dist):
    exp = math.e**((-x + mean) * dist)
    sigmoid = 1/(1 + exp)
    if sigmoid > 0.1:
        routine_immunization = (math.sin(x) + 5) / 6
        max_vacc = 0.9
        if len(sys.argv) > 15:
            max_vacc = float(sys.argv[15])
        return min(max_vacc, sigmoid * routine_immunization)
    return 0


def bell_curve(mean, dist, low, high):
    while True:
        selected_value = low + (random.random() * (high - low))
        exp_func = ((selected_value - dist)**2) / (2 * mean**2)
        dist_value = (1 / (mean * math.sqrt(2 * math.pi)) ) * (math.e**exp_func)  
        if dist_value > random.random():
            return selected_value

        
# Agents move and then are infected based off of distance to infected agents
def timestep(agents, virus_spreading, params, timestep, fear):
    agent_list = [[agent.get_coords(), agent.infection_status] for agent in agents]   
    for agent in agents:
        if params[11]: 
            agent.move(params[0], agents, fear, timestep)
        else:
            agent.move(params[0], agents, fear)

    if virus_spreading ==  True:
        for agent in agents:
            vaccination_odds = vaccination(timestep, params[7], 0.025)
            if (1 - agent.get_intelligence()) < vaccination_odds:
                agent.infection(agent_list, params[2], params[3], params[6]/10)
            else:
                agent.infection(agent_list, params[2], params[3], params[6])
            
        

#generates ids and then groups agents into families by assigning the same id
#to nearby agents
def generate_families(new_agents, agents):
    for agent in new_agents:
       for other_agent in agents:
           rand = random.random()
           other_coords = other_agent.get_coords()
           self_coords = agent.get_coords()
           distance = math.sqrt( ((self_coords[0] - other_coords[0])  ** 2 ) \
                                 + ((self_coords[1] - other_coords[1]) ** 2) )
           family_odds = 2**(-distance*0.15) 
           if rand < family_odds:
               agent.set_family_id(other_agent.get_id())
               break

           
def vital_dynamics(hotspot_maps, params):
    global agents_list
    new = new_agents(hotspot_maps, round(len(agents_list) * params[8]), params[2], params[1], False)

    generate_families(new, agents_list + new)
    for agent in new:
        agent.find_home(agents_list + new)
    
    agents_list += new
    virus_death(params[10], params[2])
    natural_death(params[9])

    
def virus_death(fatality_rate, infection_time):
    global agents_list
    infected_agents = [agent for agent in agents_list if agent.infection_status > 0]
    random.shuffle(infected_agents)

    dels = 0
    for agent in range(len(infected_agents)):
        if agent - dels > len(infected_agents) or dels >= round((fatality_rate * len(infected_agents)) / infection_time):
            break
        
        del agents_list[agent - dels]
        dels += 1
    

        
def natural_death(death_rate):
    global agents_list
    random.shuffle(agents_list)

    dels = 0
    for agent in range(len(agents_list)):
        if agent - dels > len(agents_list) or dels >= round(death_rate * len(agents_list)):
            break
        
        del agents_list[agent - dels]
        dels += 1


def new_agents(hotspot_maps, population, infection_time, map_size, new):
    if new == True:
        generated_agents = [Agent(hotspot_maps[random.randint(0, 4)], map_size, random.random(), \
                            bell_curve(0.5, 0.5, 0, 1), infection_time + 75) for i in range(5)]
        generated_agents += [Agent(hotspot_maps[random.randint(0, 4)], map_size, random.random(), \
                         bell_curve(0.5, 0.5, 0, 1)) for i in range(population - 5)]
        return generated_agents
    else:
        return [Agent(hotspot_maps[random.randint(0, 4)], map_size, random.random(), \
                      bell_curve(0.5, 0.5, 0, 1)) for i in range(population)]                 


#generates a graphs a hotspot map through pnoise, normalizes value to 0 - 1
def hotspot_generation(map_size, pers, octa):
    width = map_size + 5
    height = map_size + 5
    scale = 100.0
    octaves = round(octa)
    persistence = pers
    lacunarity = 2
    terrain = np.zeros((height, width))
    seed = random.randint(1, 100)
    
    for i in range(height):
        for j in range(width):
            terrain[i][j] = pnoise2(i / scale,
                                j / scale,
                                octaves=octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                repeatx=width,
                                repeaty=height,
                                    base=seed)
        
    # Normalize terrain values to -1–1
    normalized_terrain = ( (terrain - terrain.min()) / (terrain.max() - terrain.min()) *2) - 1
    # print(hotspot_map(normalized_terrain))
    return [list(sublist) for sublist in list(normalized_terrain)]

def hotspot_map(terrain):
    plt.imshow(terrain, cmap='terrain')
    plt.title("Colored Hotspot Map")
    plt.colorbar()
    plt.show()   


recieve_inputs()


