# simplification_photonic_cluster_state_generation_Clifford_circuit
This repository contains the python implementation for simplification photonic cluster state generation Clifford circuit

The main functions is included in the file "Optimization_functions.py", which is build upon the "GraphiQ" package that can be found on: https://github.com/graphiq-dev/graphiq

The programme uses two different ways for the simplication, the first one is a package called "qiskit" developped by IBM, the optimization transpile function, for detail, please see https://docs.quantum.ibm.com/guides/set-optimization.

The second one is a package called "pytket" developped by Quantinuum, the CliffordSimp function. Which is an optimization pass that performs a number of rewrite rules for simplifying Clifford gate sequences, similar to Duncan & Fagan (https://arxiv.org/abs/1901.10114).

The main function in "Optimization_functions.py" is "Number_of_CNOT_before_after_optimized_with_pytket" and "Number_of_CNOT_before_after_optimized_with_qiskit". They take an input of a CircuitDAG object from "GraphiQ" package and first find all the generalized emitter blocks, and apply simplification on each emitter blocks. Then output the number of CNOT gates before and after the optimization.
