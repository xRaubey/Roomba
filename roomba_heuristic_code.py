from time import perf_counter
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import animation
from queue_search_code import *

import random

import matplotlib.pyplot as plt



# WALL, CHARGER, CLEAN, DIRTY = list(range(4))

# CLEAN = 0
# WALL = 1
# ...
# DIRTYx < -x
# DIRTY_max = 

DIRTY_max = 2
CLEAN, WALL, CHARGER, CARPET = list(range(4))
# -1 = DIRTY1
# -2 = DIRTY2




SIZEL = [6,7,8,9,10]

class RoombaDomain:
    def __init__(self, pattern, size):

        SIZE = size
        # deterministic grid world
        num_rows, num_cols = SIZE, SIZE
        grid = CLEAN*np.ones((num_rows, num_cols), dtype=int)
        if(pattern == 0):
            grid[SIZE//2, 1:SIZE-1] = WALL
            grid[1:SIZE//2+1,SIZE//2] = WALL
        elif(pattern == 1):
            grid[1:SIZE-1, SIZE//2] = WALL
            grid[SIZE//2, 1:SIZE//2+1] = WALL
        elif(pattern == 2):
            grid[1:SIZE-1, SIZE//2] = WALL
            grid[SIZE//2, SIZE//2+1:SIZE-1] = WALL
        elif(pattern == 3):
            grid[SIZE//2, 1:SIZE-1] = WALL
            grid[SIZE//2+1:SIZE-1,SIZE//2] = WALL
        else:
            grid[SIZE//2, 1:SIZE-1] = WALL
            grid[1:SIZE-1,SIZE//2] = WALL
        # grid[0,0] = CHARGER
        # grid[0,-1] = CHARGER
        # grid[-1,SIZE//2] = CHARGER

        grid[0,0] = CHARGER
        grid[0,-1] = CHARGER
        grid[-1,0] = CHARGER
        grid[-1,-1] = CHARGER

        max_power = 2*SIZE + 1
        # max_power = 1
        
        
        self.grid = grid
        self.max_power = max_power


    def pack(self, g, r, c, p):
        return (g.tobytes(), r, c, p)
    def unpack(self, state):
        grid, r, c, p = state
        grid = np.frombuffer(grid, dtype=int).reshape(self.grid.shape).copy()
        return grid, r, c, p

    # def initial_state(self, roomba_position, dirty_positions):
    #     r, c = roomba_position
    #     grid = self.grid.copy()
    #     for dr, dc in dirty_positions: grid[dr, dc] = -1 * random.randint(1, DIRTY_max)
    #     return self.pack(grid, r, c, self.max_power)

    def initial_state(self, roomba_position, dirty_num, carpet_num):
        r, c = roomba_position
        grid = self.grid.copy()

        dirty_positions = np.random.permutation(list(zip(*np.nonzero(domain.grid == CLEAN))))[:dirty_num]
    
        for dr, dc in dirty_positions: grid[dr, dc] = -1 * random.randint(1, DIRTY_max)

        carpet_positions = np.random.permutation(list(zip(*np.nonzero(grid == CLEAN))))[:carpet_num]

        for dr, dc in carpet_positions: grid[dr, dc] = CARPET


        return self.pack(grid, r, c, self.max_power)

    def render(self, ax, state, x=0, y=0):
        grid, r, c, p = self.unpack(state)
        num_rows, num_cols = grid.shape
        ax.imshow(grid, cmap='gray', vmin=0, vmax=3, extent=(x-.5,x+num_cols-.5, y+num_rows-.5, y-.5))
        for col in range(num_cols+1): pt.plot([x+ col-.5, x+ col-.5], [y+ -.5, y+ num_rows-.5], 'k-')
        for row in range(num_rows+1): pt.plot([x+ -.5, x+ num_cols-.5], [y+ row-.5, y+ row-.5], 'k-')
        pt.text(c-.25, r+.25, str(p), fontsize=24)
        pt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    def valid_actions(self, state):

        # r, c is the current row and column of the roomba
        # p is the current power level of the roomba
        # grid[i,j] is WALL, CHARGER, CLEAN or DIRTY to indicate status at row i, column j.
        grid, r, c, p = self.unpack(state)
        num_rows, num_cols = grid.shape
        actions = []

        ### TODO: Update the list of valid actions as described in the instruction PDF
        # actions[k] should have the form ((dr, dc), step_cost) for the kth valid action
        # where dr, dc are the change to roomba's row and column position

        step_cost = 2 if grid[r,c] == CARPET else 1
        # grid[r,c] == CARPET

        # if grid[r,c] != WALL: actions.append(((0, 0), 1))
        # # Stay put
        # if r > 0 and grid[r-1,c] != WALL: actions.append(((-1, 0), 1))
        # # Go up
        # if r < num_rows-1 and grid[r+1,c] != WALL: actions.append(((1, 0), 1))
        # # Go down
        # if c > 0 and grid[r,c-1] > 0: actions.append(((0, -1), 1))
        # # Go left
        # if c < num_cols-1 and grid[r,c+1] != WALL: actions.append(((0, 1), 1))
        # # Go right
        # if r > 0 and c > 0 and grid[r-1,c-1] != WALL: actions.append(((-1, -1), 1))
        # # Go up left
        # if r > 0 and c < num_cols-1 and grid[r-1,c+1] != WALL: actions.append(((-1, 1), 1))
        # # Go up right
        # if r < num_rows-1 and c > 0 and grid[r+1,c-1] != WALL: actions.append(((1, -1), 1))
        # # Go down left
        # if r < num_rows-1 and c < num_cols-1 and grid[r+1,c+1] != WALL: actions.append(((1, 1), 1))
        # # Go down right

        # if p == 0: actions = [((0, 0), 1)]

        if grid[r,c] != WALL: actions.append(((0, 0), 1))
        # Stay put
        if r > 0 and grid[r-1,c] != WALL: actions.append(((-1, 0), step_cost))
        # Go up
        if r < num_rows-1 and grid[r+1,c] != WALL: actions.append(((1, 0), step_cost))
        # Go down
        if c > 0 and grid[r,c-1] > 0: actions.append(((0, -1), step_cost))
        # Go left
        if c < num_cols-1 and grid[r,c+1] != WALL: actions.append(((0, 1), step_cost))
        # Go right
        if r > 0 and c > 0 and grid[r-1,c-1] != WALL: actions.append(((-1, -1), step_cost))
        # Go up left
        if r > 0 and c < num_cols-1 and grid[r-1,c+1] != WALL: actions.append(((-1, 1), step_cost))
        # Go up right
        if r < num_rows-1 and c > 0 and grid[r+1,c-1] != WALL: actions.append(((1, -1), step_cost))
        # Go down left
        if r < num_rows-1 and c < num_cols-1 and grid[r+1,c+1] != WALL: actions.append(((1, 1), step_cost))
        # Go down right

        if p == 0: actions = [((0, 0), 1)]
        if p == 1 and grid[r,c] == CARPET: actions = [((0, 0), 1)]


        return actions
    
    def perform_action(self, state, action):
        grid, r, c, p = self.unpack(state)
        dr, dc = action

        # TODO: update grid, r, c, and p as described in the instruction PDF

        # if dr!=0 and dc!=0: p = p-1

        step_cost = 2 if grid[r,c] == CARPET else 1

        # print("sc",step_cost)

        if dr!=0 or dc!=0: 
            p = p - step_cost
        
        r, c = r + dr, c + dc
        if grid[r,c] == CHARGER and dr==0 and dc==0: p = p+1
        
        # if grid[r,c] == DIRTY and dr==0 and dc==0 and p>0: 
        #     grid[r,c] = CLEAN
        #     p=p-1


        # Clean Dirty Level by 1
        if grid[r,c] < CLEAN and dr==0 and dc==0 and p>0: 
            grid[r,c] = grid[r,c]+1
            p=p-1



        new_state = self.pack(grid, r, c, p)
        return new_state

    def is_goal(self, state):
        grid, r, c, p = self.unpack(state)

        # # In a goal state, no grid cell should be dirty
        # result = (grid != DIRTY).all()


        # 2222222
        # In a goal state, no grid cell should be dirty
        result = (grid >= CLEAN ).all()




        ### TODO: Implement additional requirement that roomba is back at a charger

        result2= grid[r,c] == CHARGER

        return result and result2

    def is_game_over(self, state, steps_left):

        if steps_left == 0:
            return True

        if(self.is_goal(state)):
            return True

        grid, r, c, p = self.unpack(state)

        if(p == 0 or (p==1 and grid[r,c] == CARPET)):
            return True

        return False
    def get_dirtiness(self, state):
        grid, r, c, p = self.unpack(state)
        return grid[grid < CLEAN].sum()    
    

    def get_game_score(self, state, initial_state):
        grid, r, c, p = self.unpack(state)
        igrid, ir, ic, ip = self.unpack(initial_state)

        i_dirty = igrid[igrid < CLEAN].sum()
        final_dirty = grid[grid < CLEAN].sum()

        return (final_dirty - i_dirty) 

    def simple_heuristic(self, state):
        grid, r, c, p = self.unpack(state)

        # get list of dirty positions
        # dirty[k] has the form (i, j)
        # where (i, j) are the row and column position of the kth dirty cell
        # dirty = list(zip(*np.nonzero(grid == DIRTY)))

        # 33333
        dirty = list(zip(*np.nonzero(grid < CLEAN)))

        # if no positions are dirty, estimate zero remaining cost to reach a goal state
        if len(dirty) == 0: return 0

        # otherwise, get the distance from the roomba to each dirty square
        dists = [max(np.fabs(dr-r), np.fabs(dc-c)) for (dr, dc) in dirty]

        # estimate the remaining cost to goal as the largest distance to a dirty position
        return int(max(dists))

    def better_heuristic(self, state):

        ### TODO: Implement a "better" heuristic than simple_heuristic
        # "Better" means more memory-efficient (fewer popped nodes during A* search)

        grid, r, c, p = self.unpack(state)
        # dirty = list(zip(*np.nonzero(grid == DIRTY)))

        #44444
        dirty = list(zip(*np.nonzero(grid < CLEAN)))

        chargers = list(zip(*np.nonzero(grid == CHARGER)))

        if len(dirty) == 0: return 0

        curr_r=r
        curr_c=c
        farest = 0

        for (dr, dc) in dirty:
            curr = max(np.fabs(dr-r), np.fabs(dc-c))
            if farest<=curr: 
                farest=curr
                curr_r=dr
                curr_c=dc
        mindist_charger1 = min([max(np.fabs(dr-curr_r), np.fabs(dc-curr_c)) for (dr, dc) in chargers])
        farest = farest+mindist_charger1  
        return farest     

if __name__ == "__main__":

    pattern = int(input("Wall Pattern (0-4): "))
    size = int(input("Type in size (3-15): "))
    dirty_num = int(input("Number of dirty squares (max:5): "))
    carpet_num = int(input("Number of carpet: "))
    r = int(input("Roomba position r =  "))
    c = int(input("Roomba position c = "))

    mode = str(input("Choose a mode: user, ai_random, ai_tree: "))
    while(mode != 'user' and mode != 'ai_random' and mode != 'ai_tree'):
        mode = str(input("Wrong Mode. Choose a mode: user, ai_random, ai_tree: "))

    if(size>=15):
            size = 15
    elif(size<=3):
            size = 3

    if(r<0 or r>size-1):
        r=0
    if(c<0 or c>size-1):
        c=0

    if(dirty_num > 5 ):
        dirty_num = 5

    # set up initial state by making five random open positions dirty
    domain = RoombaDomain(pattern, size)

    # dp = input("Dirty Position")
    

    # init = domain.initial_state(
    #     roomba_position = (0, 0),
    #     dirty_positions = np.random.permutation(list(zip(*np.nonzero(domain.grid == CLEAN))))[:dirty_num])

    # 5555
    init = domain.initial_state(
        roomba_position = (r, c),
        dirty_num = dirty_num, 
        carpet_num = carpet_num
        )
    problem = SearchProblem(domain, init, domain.is_goal)
    
    start = perf_counter()
    plan, node_count = a_star_search(problem, domain.simple_heuristic)
    astar_time = 0
    perf_counter() - start
    print("astar_time", astar_time)
    print("node count", node_count)

    # print('plan', plan)
    

    max_step = 2*len(plan)

    # start = perf_counter()
    # plan, node_count = a_star_search(problem, domain.better_heuristic)
    # astar_time = perf_counter() - start
    # print("better heuristic:")
    # print("astar_time", astar_time)
    # print("node count", node_count)

    game_over = False
    print('Game Start: ')
    states = [problem.initial_state]
    g,r,c,p = domain.unpack(problem.initial_state)
    print("Grid:\n",g,"\nr:",r, " c:",c, "\np:",p)
    input("Press Enter to continue...")

    while (not game_over):
        # reconstruct the intermediate states along the plan
        if mode == 'ai_tree':
            for a in range(len(plan)):
                s = domain.perform_action(states[-1], plan[a])
                states.append(s)
                g,r,c,p = domain.unpack(s)
                print("Grid:\n",g,"\nr:",r, " c:",c, "\np:",p)
                input("Press Enter to continue...")
                max_step -= 1
                game_over = domain.is_game_over(s,max_step)
        elif mode == 'user':
            dr = int(input("Please Enter Movement for r (-1, 0 , 1): "))
            dc = int(input("Please Enter Movement for c (-1, 0 , 1): "))
            valid_actions = domain.valid_actions(states[-1])
            va = False
            for a,b in valid_actions:
                if((dr, dc) == a):
                    va =True
                    break
            if(not va):
                print('Not a Valid Action!!')
                continue

            s = domain.perform_action(states[-1], (dr,dc))
            states.append(s)
            g,r,c,p = domain.unpack(s)
            print("Grid:\n",g,"\nr:",r, " c:",c, "\np:",p)
            input("Press Enter to continue...")
            max_step -= 1
            game_over = domain.is_game_over(s, max_step)
        elif mode == 'ai_random':
            valid_actions = domain.valid_actions(states[-1])
            a,b = random.choice(valid_actions)
            s = domain.perform_action(states[-1], a)
            states.append(s)
            g,r,c,p = domain.unpack(s)
            print("Grid:\n",g,"\nr:",r, " c:",c, "\np:",p)
            input("Press Enter to continue...")
            max_step -= 1
            game_over = domain.is_game_over(s, max_step)


    print('Game is over!!')
    print('Your Score is: ')
    print(domain.get_game_score(states[-1], problem.initial_state))

    ''''
    for s in states:
        g,r,c,p = domain.unpack(s)
        print("Grid:\n",g,"\nr:",r, " c:",c, "\np:",p)
   '''
   
    def run_ex_by_id (domain, size, id):
        experiment_size = size
        iterator = 100
        node_processed = []
        final_score = []
        final_score_baseline = []
        id += 1
        while(iterator>0):
            r = random.randint(0,experiment_size-1)
            c = random.randint(0,experiment_size-1)
            dirty_num = random.randint(0,5)
            if pow(experiment_size,2)-dirty_num-4 > 0:
                carpet_num = random.randint(1,pow(experiment_size,2)-dirty_num-4)
            else:
                carpet_num = 1

            init_experiment = domain.initial_state(
            roomba_position = (r, c),
            dirty_num = dirty_num, 
            carpet_num = carpet_num
            )

            problem_experiment = SearchProblem(domain, init_experiment, domain.is_goal)
            plan, node_count = a_star_search(problem_experiment, domain.simple_heuristic)
            node_processed.append(node_count)

            # Baseline
            max_step = 2*len(plan)
            game_over = False
            print('Experiment ', id ,' (baseline): ', 100-iterator)
            states = [problem_experiment.initial_state]
            g,r,c,p = domain.unpack(problem_experiment.initial_state)
            while (not game_over):
                valid_actions = domain.valid_actions(states[-1])
                a,b = random.choice(valid_actions)
                s = domain.perform_action(states[-1], a)
                states.append(s)
                max_step -= 1
                game_over = domain.is_game_over(s, max_step)
                if max_step == 0:
                    break
            final_score_baseline.append(domain.get_game_score(states[-1], problem_experiment.initial_state))

            # Tree based AI
            max_step = 2*len(plan)
            game_over = False
            print('Experiment ', id ,' (Tree AI): ', 100-iterator)
            states = [problem_experiment.initial_state]
            g,r,c,p = domain.unpack(problem_experiment.initial_state)

            while (not game_over):
                for a in range(len(plan)):
                    s = domain.perform_action(states[-1], plan[a])
                    states.append(s)
                    max_step -= 1
                    game_over = domain.is_game_over(s,max_step)
                if max_step == 0:
                    break
            final_score.append(domain.get_game_score(states[-1], problem_experiment.initial_state))

            iterator -= 1

        print('Experiment ', id ,': Final score list (baseline): ',final_score_baseline)
        print('Experiment ', id ,': Final score list (Tree-based AI): ',final_score)

        return (node_processed, final_score_baseline, final_score)


        # hist1 = plt.figure(1)
        # plt.hist(node_processed, label= '# of nodes processed')
        # plt.xlabel('Number of nodes processed')
        # plt.ylabel('Amount')

        # hist2 = plt.figure(2, label= 'score')
        # plt.hist(final_score)
        # plt.xlabel('Score')
        # plt.ylabel('Amount')

        # plt.show()
   
    run_experiment = str(input('Run experiments? Y/N: '))
    if(run_experiment == 'Y' or run_experiment == 'y' or run_experiment == ''):
        experiment = [(0,3),(1,5),(2,7),(3,10),(4,15)]
        result_set = []
        for i in range(5):
            pattern, size = experiment[0]
            domain = RoombaDomain(pattern, size)
            print('- Domain: ',i,' - Patter: ',pattern, ' - Size: ', size )
            result_set.append(run_ex_by_id(domain, size, i))

        for i in range(5):
            node_processed, final_score_baseline, final_score = result_set[i]
            plt.subplot(5,3,i*3+1)
            plt.hist(final_score_baseline, label= 'score (baseline)', bins=max(final_score_baseline)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Score (baseline)')
            plt.ylabel('Amount')

            plt.subplot(5,3,i*3+2)
            plt.hist(final_score, label= 'score (Tree-based AI)', bins=max(final_score)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Score (Tree-based AI)')
            plt.ylabel('Amount')

            plt.subplot(5,3,i*3+3)
            plt.hist(node_processed, label= '# of nodes processed (Tree-based AI)', bins=max(node_processed)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Number of nodes processed')
            plt.ylabel('Amount')
        
        plt.show()
        
    print('done')
   
   
    # Animate the plan

    # fig = pt.figure(figsize=(8,8))

    # def drawframe(n):
    #     pt.cla()
    #     domain.render(pt.gca(), states[n])

    # # blit=True re-draws only the parts that have changed.
    # anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=500, blit=False)
    # pt.show()

