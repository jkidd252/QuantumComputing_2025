# QuantumComputing_2025
Group 5 - Code for 'Quantum Computing' project 2025

## ℹ️ Overview

This directory contains all the code produced by group 5 for the Quantum Computing Project.

Code is written in an OOP style and operations are executed by running files whose name starts with 'main_', these include:
> main_GroverType_comparision.py

This runs sparse and normal versions of the gate and target based grovers algorithm, this file produces an image 'time_vs_qubit.png' which is shown in Figure 16 in the report.

> main_GFT_final.py

This runs the QFT algorithm for phase and order estimation. 

> main_MatrixProd_Performance.py

This runs both sparse matrix and regular matrix versions of the matrix product and produces an image 'Logstep.png' which shows the time needed to compute the product as a function of the sparsity of the input matrix. The resulting image is shown in Figure 12 in the report.


## ⬇️ Installation instructions

The various main files listed above have dependencies on other files given in this repository. It is recomended to download all the files within this repository before executing a main file, where each file must be contained within the same directory for proper execution of the code.

Each main file features a number of parameters that can be changed, for instance to change the number of qubits being input to the QFT, these are clearly highlighted in each file. The parameters currently set are those used to produce the various figures and results stated in our report.
Each function used is docstringed and supports the python 'help()' command, these can be used to trouble shoot issues with the code.

Each file can be run from the command line using the following command (this is standard execution of a python file, it is included here only for completeness).
```bash
python main_'FILENAME'.py
```
