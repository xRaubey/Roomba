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
  - Type the following commands in the command prompt if your system doesn't have pip installed.
   - ```sudo apt install python-pip``` or ```sudo3 apt install python-pip```
- Installing NumPy
  - Type the following commands in the command prompt if your system doesn't have Numpy installed.
   - ```pip install numpy```
- Installing PyTorch
  - Type the following commands in the command prompt if your system doesn't have PyTorch installed.
   - ```pip install torch``` or ```pip3 install torch```
- Install matplotlib
  - Type the following commands in the command prompt if your system doesn't have matplotlib installed.
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
- Choose the way to run the program (user, ai_random, ai_tree, ai_NN1, ai_NN2).
  - user mode: For each step, press enter to continue, enter -1, 0, 1 for roomba’s next row position and column position.
  - ai_random mode: For each step, press enter to continue.
  - ai_tree mode: For each step, press enter to continue.
  - ai_NN1 mode: For each step, press enter to continue.
  - ai_NN2 mode: For each step, press enter to continue.
- Choose whether to run 5 predefined experiments (and 100 games using neural network) or not.
  - Note that it may take some time to generate histograms for the result of experiments.

# How to Run the Computer Experiments?
Right after the interactive domain program has finished and the final score has shown in terminal, a command prompt “Run experiments? ” will appear. Type “Y”, “y” or press enter to run experiments automatically.

# Attribution
HW2 Problem 7 Source Code, CIS 667: Introduction to Artificial Intelligence (Fall 2022), Professor Garrett Katz, Syracuse University

# Team Members
- Yuqing Yang (yyang74@syr.edu)
- Shao-Peng Yang (syang32@syr.edu)

# Bibliography
- [1]  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
- [2] Harris, C.R., Millman, K.J., van der Walt, S.J. et al(2020). Array programming with NumPy. Nature 585, 357–362. DOI: 10.1038/s41586-020-2649-2. (Publisher link).
- [3] P. E. Hart, N. J. Nilsson and B. Raphael(1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. In IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100-107, July 1968, doi: 10.1109/TSSC.1968.300136.
- [4]Russell, S. J., & Norvig, P. (2020). Artificial intelligence: a modern approach 4th Edition. Englewood Cliffs, N.J., Prentice Hall.
- [5] Katz, Garrett. CIS 667: Introduction to Artificial Intelligence Homework 1 Code.
- [6] P. F. Baldi and K. Hornik(1995). Learning in linear neural networks: a survey. in IEEE Transactions on Neural Networks, vol. 6, no. 4, pp. 837-858, July 1995, doi: 10.1109/72.392248.

