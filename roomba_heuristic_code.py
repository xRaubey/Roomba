from time import perf_counter
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import animation
from queue_search_code import *

import random

import matplotlib.pyplot as plt
import torch as tr

# NN_size = 7
OBJ_amount = 7

DATA_SIZE = 1000

class LinNet(tr.nn.Module):
    def __init__(self, li_size, hid_features):
        super(LinNet, self).__init__()
        self.to_hidden = tr.nn.Linear(OBJ_amount*li_size**2, hid_features)
        self.to_output = tr.nn.Linear(hid_features, 1)

        # self.to_output = tr.nn.Linear(OBJ_amount*size**2, 1)

    def forward(self, x):
        h = tr.tanh(self.to_hidden(x.reshape(x.shape[0],-1)))
        y = tr.relu(self.to_output(h))

        # y = tr.relu(self.to_output(x.reshape(x.shape[0],-1)))
        return y

class ConvNet(tr.nn.Module):
    def __init__(self, cv_size, hid_features):
        super(ConvNet, self).__init__()
        self.to_hidden = tr.nn.Conv2d(OBJ_amount, hid_features, 2)
        self.to_output = tr.nn.Linear(hid_features*(cv_size-1)**2, 1)
    def forward(self, x):
        # h = tr.relu(self.to_hidden(x))
        # y = tr.tanh(self.to_output(h.reshape(x.shape[0],-1)))
        h = tr.tanh(self.to_hidden(x))
        y = tr.relu(self.to_output(h.reshape(x.shape[0],-1)))
        return y


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
    
    def __init__(self, pattern, rb_size, net = None, net2 = None):
        self.roomba_size = rb_size
        SIZE = self.roomba_size
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

        grid[0,0] = CHARGER
        grid[0,-1] = CHARGER
        grid[-1,0] = CHARGER
        grid[-1,-1] = CHARGER

        max_power = 2*SIZE + 1        
        
        self.grid = grid
        self.max_power = max_power

        self.ai_net = net
        self.ai_net2 = net2


    def pack(self, g, r, c, p):
        return (g.tobytes(), r, c, p)
    def unpack(self, state):
        grid, r, c, p = state
        grid = np.frombuffer(grid, dtype=int).reshape(self.grid.shape).copy()
        return grid, r, c, p

    def initial_state(self, roomba_position, dirty_num, carpet_num):
        r, c = roomba_position
        grid = self.grid.copy()

        dirty_positions = np.random.permutation(list(zip(*np.nonzero(self.grid == CLEAN))))[:dirty_num]
    
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

        step_cost = 2 if grid[r,c] == CARPET else 1

        if dr!=0 or dc!=0: 
            p = p - step_cost
        
        r, c = r + dr, c + dc
        if grid[r,c] == CHARGER and dr==0 and dc==0: p = p+1
        
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

    def encode1_AI(self,state):
        symbol = np.array([-2,-1,0,1,2,3]).reshape(-1,1,1)
        onehot = (symbol == state).astype(np.float32)
        return tr.tensor(onehot)

    def encode2_AI(self,state):
        symbol = np.array([4]).reshape(-1,1,1)
        onehot = (symbol == state).astype(np.float32)
        return tr.tensor(onehot)


    def encode_AI(self,state1, state2):
        tensor = tr.cat((self.encode1_AI(state1),self.encode2_AI(state2)),0)
        return tensor

    def AI_heuristic(self, state):
        
        if self.ai_net!=None:
            grid, r, c, p = self.unpack(state)
            rp_grid = np.zeros([self.roomba_size,self.roomba_size])
            
            rp_grid[r,c] = 4
            x = self.encode_AI(grid, rp_grid).unsqueeze(0)
            y = self.ai_net(x)
            return y
        else:
            print('No Net1 Detected')
            return 0


if __name__ == "__main__":


    # --- User Inputs ---

    pattern = int(input("Wall Pattern (0-4): "))
    # print('Please use small size for NN part.')
    size = int(input("Type in size (3-15): "))
    dirty_num = int(input("Number of dirty squares (max:5): "))
    carpet_num = int(input("Number of carpet: "))
    r = int(input("Roomba position r =  "))
    c = int(input("Roomba position c = "))

    NN_size = size
    user_input_size = size

    mode = str(input("Choose a mode: user, ai_random, ai_tree, ai_NN1, ai_NN2: "))
    while(mode != 'user' and mode != 'ai_random' and mode != 'ai_tree' and mode != 'ai_NN1' and mode != 'ai_NN2'):
        mode = str(input("Wrong Mode. Choose a mode: user, ai_random, ai_tree: "))

    if(size>=15):
            size = 15
            user_input_size = size
            NN_size = size
    elif(size<=3):
            size = 3
            user_input_size = size
            NN_size = size

    if(r<0 or r>user_input_size-1):
        r=0
    if(c<0 or c>user_input_size-1):
        c=0

    if(dirty_num > 5 ):
        dirty_num = 5


    # --- Generate Training Set and Testing Set ---


    def example_error(net, example):
        state, heuristic, roomba_position, roomba_grid = example
        x = encode(state, roomba_grid).unsqueeze(0)
        y = net(x)
        e = (y - heuristic)**2
        return e


    def random_state():

        # set the size here for NN ai
        nn_size = NN_size
        r = random.randint(0,nn_size-1)
        c = random.randint(0,nn_size-1)
        pattern = random.randint(0,4)
        dirty_num = random.randint(1,5)

        charger_amount = 4
        wall_amount = (2*(nn_size-2))-1

        if pow(nn_size,2)-dirty_num-charger_amount-wall_amount >= 0:
            carpet_num = random.randint(0,pow(nn_size,2)-dirty_num-charger_amount-wall_amount)
        else:
            carpet_num = 0
        state = (pattern,r,c,dirty_num,carpet_num,nn_size)
        return state

    def generate_dataset():
        state_cost = []

        # size of dataset, should be 1000
        while(len(state_cost)<DATA_SIZE):

            pattern,r,c,dirty_num,carpet_num,size_rs = random_state()

            domain = RoombaDomain(pattern, size_rs)
            
            example = domain.initial_state(
            roomba_position = (r, c),
            dirty_num = dirty_num, 
            carpet_num = carpet_num
            )
            
            problem_example = SearchProblem(domain, example, domain.is_goal)
            plan,count = a_star_search(problem_example,domain.simple_heuristic)

            states = [problem_example.initial_state]
            g,r,c,p = domain.unpack(problem_example.initial_state)

            actual_cost = 0
            current_cost = [0]

            grids = [g]
            roomba_position = [(r,c)]

            rp_grid = np.zeros([size_rs,size_rs])
            rp_grid[r,c] = 4
            rp_grids = [rp_grid]

            for a in range(len(plan)):

                if(g[r,c]==CARPET and plan[a] != (0,0)): 
                    actual_cost += 2
                else: 
                    actual_cost += 1

                s = domain.perform_action(states[-1], plan[a])
                states.append(s)
                g,r,c,p = domain.unpack(s)

                current_cost.append(actual_cost)
                grids.append(g)
                roomba_position.append((r,c))
                rp_grid = np.zeros([size,size])
                rp_grid[r,c] = 4
                rp_grids.append(rp_grid)
                
            heuristic_list = [actual_cost-x for x in current_cost]
            result_list = zip(grids,heuristic_list,roomba_position,rp_grids)
            result_list = list(result_list)
            state_cost.extend(result_list)
            print(len(state_cost),'/ 1000')
        return state_cost

    # -2 dirty level 2, -1 dirty level 1, 0 empty, 1 wall, 2 charger, 3 carpet, 4 roomba.
    def encode1(state):
        symbol = np.array([-2,-1,0,1,2,3]).reshape(-1,1,1)
        onehot = (symbol == state).astype(np.float32)
        return tr.tensor(onehot)

    def encode2(state):
        symbol = np.array([4]).reshape(-1,1,1)
        onehot = (symbol == state).astype(np.float32)
        return tr.tensor(onehot)


    def encode(state1, state2):
        tensor = tr.cat((encode1(state1),encode2(state2)),0)
        return tensor


    print('Generating Training Set...')
    training_examples = generate_dataset()

    print('Generating Testing Set...')
    testing_examples = generate_dataset()


     # -- Neural Netwok (Tree + NN) ---


    def batch_error(net, batch):
        states, utilities = batch
        u = utilities.reshape(-1,1).float()
        y = net(states)
        e = tr.sum((y - u)**2) / utilities.shape[0]
        return e

    # NN configuration 1  -- Learning Rate = 0.001 and hid_features = 16
    if(mode == 'ai_NN1'):
        net = LinNet(li_size=NN_size, hid_features=16)
        optimizer = tr.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # NN configuration 2 -- Learning Rate = 0.0005 and hid_features = 32
    elif(mode == 'ai_NN2'):
        net = LinNet(li_size=NN_size, hid_features=32)
        optimizer = tr.optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
    else:
        net = LinNet(li_size=NN_size, hid_features=16)
        optimizer = tr.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    states, utilities, roomba_positions, roomba_grids = zip(*training_examples)
    training_batch = tr.stack(tuple(map(encode, states,roomba_grids))), tr.tensor(utilities)

    states, utilities, roomba_positions, roomba_grids = zip(*testing_examples)
    testing_batch = tr.stack(tuple(map(encode, states, roomba_grids))), tr.tensor(utilities)

    curves = [], []


    for epoch in range(50000):
        
        optimizer.zero_grad()

        training_error, testing_error = 0, 0
        
        # Not batched
        # for n, example in enumerate(training_examples):
        #     # print('n= ',n)
        #     e = example_error(net, example)
        #     e.backward()
        #     training_error += e.item()
        # training_error /= len(training_examples)


        # Batched

        e = batch_error(net, training_batch)
        e.backward()
        training_error = e.item()

        with tr.no_grad():
            e = batch_error(net, testing_batch)
            testing_error = e.item()

        optimizer.step()
        if epoch % 1000 == 0:
        # if epoch % 100 == 0:
            print("%d: %f, %f" % (epoch, training_error, testing_error))
        # curves.append(training_error)
        curves[0].append(training_error)
        curves[1].append(testing_error)


    # diagram

    pt.plot(curves[0], 'b-')
    pt.plot(curves[1], 'r-')
    pt.plot()
    pt.legend(["Train","Test"])
    pt.show()

    # diagram





   # -- Game Part ---


    # set up initial state by making five random open positions dirty
    domain = RoombaDomain(pattern, user_input_size, net)

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

    plan_NN, node_count_NN = a_star_search(problem, domain.AI_heuristic)
    astar_time_NN = 0
    perf_counter() - start
    print("astar_time_NN", astar_time_NN)
    print("node count NN", node_count_NN)

    max_step = 2*len(plan)
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
        elif mode == 'ai_NN1':
            ## Tree + NN
            for a in range(len(plan_NN)):
                s = domain.perform_action(states[-1], plan_NN[a])
                states.append(s)
                g,r,c,p = domain.unpack(s)
                print("Grid:\n",g,"\nr:",r, " c:",c, "\np:",p)
                input("Press Enter to continue...")
                max_step -= 1
                game_over = domain.is_game_over(s,max_step)
        elif mode == 'ai_NN2':
            ## Tree + NN
            for a in range(len(plan_NN)):
                s = domain.perform_action(states[-1], plan_NN[a])
                states.append(s)
                g,r,c,p = domain.unpack(s)
                print("Grid:\n",g,"\nr:",r, " c:",c, "\np:",p)
                input("Press Enter to continue...")
                max_step -= 1
                game_over = domain.is_game_over(s,max_step)


    print('Game is over!!')
    print('Your Score is: ')
    print(domain.get_game_score(states[-1], problem.initial_state))

    ''''
    for s in states:
        g,r,c,p = domain.unpack(s)
        print("Grid:\n",g,"\nr:",r, " c:",c, "\np:",p)
   '''



    # --- Evaluation ---

    def run_ex_by_id (domain, ex_size, id):

        # domain =_domain

        experiment_size = ex_size
        iterator = 100
        
        node_processed = []
        final_score = []
        final_score_baseline = []
        id += 1

        while(iterator>0):
            
            r = random.randint(0,experiment_size-1)
            c = random.randint(0,experiment_size-1)
            dirty_num = random.randint(0,5)

            # dirty_num = random.randint(1,3)

            charger_amount = 4
            wall_amount = (2*(experiment_size-2))-1

            if pow(experiment_size,2)-dirty_num-charger_amount-wall_amount >= 0:
                carpet_num = random.randint(0,pow(experiment_size,2)-dirty_num-charger_amount-wall_amount)
            else:
                carpet_num = 0

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


    def run_nn (net):

        nn_size = user_input_size
        iterator = 100
        node_processed_NN = []
        final_score_NN1 = []

        charger_amount = 4
        wall_amount = (2*(nn_size-2))-1

        while(iterator>0):

            # Tree NN 1

            pattern = random.randint(0,4)
            domain = RoombaDomain(pattern,nn_size,net)

            r = random.randint(0,nn_size-1)
            c = random.randint(0,nn_size-1)
            # dirty_num = random.randint(0,5)
            dirty_num = random.randint(1,3)
            if pow(nn_size,2)-dirty_num-charger_amount-wall_amount >= 0:
                carpet_num = random.randint(0,pow(nn_size,2)-dirty_num-charger_amount-wall_amount)
            else:
                carpet_num = 0
            init_nn = domain.initial_state(
            roomba_position = (r, c),
            dirty_num = dirty_num, 
            carpet_num = carpet_num
            )

            problem_experiment = SearchProblem(domain, init_nn, domain.is_goal)
            plan_NN, node_count_NN = a_star_search(problem_experiment, domain.AI_heuristic)
            node_processed_NN.append(node_count_NN)

            max_step = 2*len(plan_NN)
            game_over = False
            if(mode == 'ai_NN1'):
                print('(Tree + NN1): ', 100-iterator)
            elif(mode == 'ai_NN2'):
                print('(Tree + NN2): ', 100-iterator)
            else:
                print('(Tree + NN): ', 100-iterator)
            states = [problem_experiment.initial_state]
            g,r,c,p = domain.unpack(problem_experiment.initial_state)

            while (not game_over):
                for a in range(len(plan_NN)):
                    s = domain.perform_action(states[-1], plan_NN[a])
                    states.append(s)
                    max_step -= 1
                    game_over = domain.is_game_over(s,max_step)
                if max_step == 0:
                    break
            final_score_NN1.append(domain.get_game_score(states[-1], problem_experiment.initial_state))


            iterator -= 1
        if(mode == 'ai_NN1'):
            print('NN1 : Final score list (AI + NN1): ',final_score_NN1)
        elif(mode == 'ai_NN2'):
            print('NN2 : Final score list (AI + NN2): ',final_score_NN1)
        else:
            print('NN : Final score list (AI + NN): ',final_score_NN1)
        return (node_processed_NN, final_score_NN1)

   
    run_experiment = str(input('Run experiments? Y/N: '))
    if(run_experiment == 'Y' or run_experiment == 'y' or run_experiment == ''):
        experiment = [(0,3),(1,5),(2,7),(3,10),(4,15)]

        result_set = []
        for i in range(5):
            pattern, ex_size = experiment[0]
            domain = RoombaDomain(pattern, ex_size)

            print('- Domain: ',i,' - Patter: ',pattern, ' - Size: ', ex_size )

            result_set.append(run_ex_by_id(domain, ex_size, i))

        node_processed_NN, final_score_NN1 = run_nn(net)

        print('Printing Histograms...')

        row = 6
        col = 3

        for i in range(5):
            # node_processed, final_score_baseline, final_score = result_set[i]
            node_processed, final_score_baseline, final_score = result_set[i]
            plt.subplot(row,col,i*col+1)
            plt.hist(final_score_baseline, label= 'score (baseline)', bins=max(final_score_baseline)-min(final_score_baseline)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Score (baseline)')
            plt.ylabel('Amount')

            plt.subplot(row,col,i*col+2)
            plt.hist(final_score, label= 'score (Tree-based AI)', bins=max(final_score)-min(final_score)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Score (Tree-based AI)')
            plt.ylabel('Amount')

            plt.subplot(row,col,i*col+3)
            plt.hist(node_processed, label= '# of nodes processed (Tree-based AI)', bins=max(node_processed)-min(node_processed)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Number of nodes processed')
            plt.ylabel('Amount')


        plt.subplot(row,col,16)
        if(mode =='ai_NN1'):
            plt.hist(final_score_NN1, label= 'score (Tree-based + NN1)', bins=max(final_score_NN1)-min(final_score_NN1)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Score (Tree-based + NN1)')
        elif(mode =='ai_NN2'):
            plt.hist(final_score_NN1, label= 'score (Tree-based + NN2)', bins=max(final_score_NN1)-min(final_score_NN1)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Score (Tree-based + NN2)')
        else:
            plt.hist(final_score_NN1, label= 'score (Tree-based + NN)', bins=max(final_score_NN1)-min(final_score_NN1)+1, align='mid' ,edgecolor='black')
            plt.xlabel('Score (Tree-based + NN)')
        plt.ylabel('Amount')

        plt.subplot(row,col,17)
        if(mode =='ai_NN1'):
            plt.hist(node_processed_NN, label= '# of nodes processed (Tree-based + NN1)', bins=(max(node_processed_NN)-min(node_processed_NN))+1, align='mid' ,edgecolor='black')
            plt.xlabel('Number of nodes processed (Tree-base + NN1)')
        elif(mode =='ai_NN2'):
            plt.hist(node_processed_NN, label= '# of nodes processed (Tree-based + NN2)', bins=(max(node_processed_NN)-min(node_processed_NN))+1, align='mid' ,edgecolor='black')
            plt.xlabel('Number of nodes processed (Tree-base + NN)')
        else:
            plt.hist(node_processed_NN, label= '# of nodes processed (Tree-based + NN)', bins=(max(node_processed_NN)-min(node_processed_NN))+1, align='mid' ,edgecolor='black')
            plt.xlabel('Number of nodes processed (Tree-base + NN)')
        plt.ylabel('Amount')


        
        plt.show()
        
    print('- Done -')
   
