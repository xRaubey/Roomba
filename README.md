# Roomba AI
CIS667-Project

# Contents of This File
- Introduction
- Required dependencies
- Run the interactive domain program
- Run the computer experiments
- Attribution
- Team members

# Introduction
Roomba AI is a project that simulates the floor cleaning procedure conducted by a roomba. Users are allowed to manage the settings such as matrix size, roomba starting position, wall positions to view the cleaning procedure and evaluate the different outcomes. In the meanwhile, users can control the roomba’s next move by typing in information manually or letting the AI do the job.

# How to Install Required Dependencies?
- Install Python
  - Type the following commands in the command prompt to check if your system does have python installed.
   ```python -v ``` or ```python3 -v ```
  - If python is not detected in your system, you should install python first.
- Installing pip
  - Type the following commands in the command prompt if your system does have matplotlib installed.
   - ```sudo apt install python-pip``` or ```sudo3 apt install python-pip```
- Installing NumPy
  - Type the following commands in the command prompt if your system does have bumpy installed.
   - ```pip install numpy```
- Install matplotlib
  - Type the following commands in the command prompt if your system does have matplotlib installed.
   - ```pip install matplotlib```

# How to Run the Interactive Domain Program?
- Open terminal.
- Run the python file “roomba_heuristic_code.py” in the project folder.
- Select wall pattern for the domain (from 0 to 4).
- Select the size of matrix for the domain (from 3*3 to 15*15).
- Select the number of dirty squares (from 0 to 5).
- Select the number of carpet (at least 0).
- Select roomba’s starting position (row).
- Select roomba’s starting position (column).
- Choose the way to run the program (user, ai_random or ai_tree).
  - user mode: For each step, press enter to continue, enter -1, 0, 1 for roomba’s next row position and column position.
  - ai_random mode: For each step, press enter to continue.
  - ai_tree mode: For each step, press enter to continue.
- Choose whether to run 5 predefined experiments or not.
  - Note that it may take some time to generate histograms for the result of experiments.

# How to Run the Computer Experiments?
Right after the interactive domain program has finished and the final score has shown in terminal, a command prompt “Run experiments? ” will appear. Type “Y”, “y” or press enter to run experiments automatically.

# Attribution
HW2 Problem 7 Source Code, CIS 667: Introduction to Artificial Intelligence (Fall 2022), Professor Garrett Katz, Syracuse University

# Team members
- Yuqing Yang (yyang74@syr.edu)
- Shao-Peng Yang (syang32@syr.edu)
