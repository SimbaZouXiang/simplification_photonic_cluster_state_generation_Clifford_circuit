"""
The following functions are built upon the graphiq library and are used to optimize the quantum circuits.
Author: XIANG ZOU
Institution: University of Toronto, Toronto, Ontario, Canada
Date: 2024-07-11
Version: 1.0
"""


import graphiq as gq
from graphiq.benchmarks.circuits import bell_state_circuit
import networkx as nx
import graphiq.circuit.ops as ops
from graphiq.circuit.circuit_dag import CircuitDAG
import numpy as np
import matplotlib.pyplot as plt

def _get_SigmaXYZ_list(circuit):
    """
    This function is used to get the SigmaX, SigmaY, and SigmaZ nodes in the circuit.

    
    """
    try:
        SigmaX_list = np.array(circuit.node_dict["SigmaX"])
    except:
        SigmaX_list = np.array([])
    try:
        SigmaY_list = np.array(circuit.node_dict["SigmaY"])
    except:
        SigmaY_list = np.array([])
    try:
        SigmaZ_list = np.array(circuit.node_dict["SigmaZ"])
    except:
        SigmaZ_list = np.array([])
        
    return SigmaX_list, SigmaY_list, SigmaZ_list

def combine_pauli_gate(circuit):
    """
    Combine the Pauli gates in the circuit

    :param circuit: CircuitDAG object
    :type circuit: CircuitDAG
    """
    flag = True
    while flag == True:
        sx_list, sy_list, sz_list = _get_SigmaXYZ_list(circuit)
        for gate in sx_list:
            for edge in circuit.edge_dict["e"]:
                if gate == edge[0]:
                    nextgate = edge[1]
                    if nextgate in sy_list:
                        circuit.remove_op(nextgate)
                        circuit.replace_op(gate, ops.SigmaY(register=0, reg_type="e"))


def _get_list_of_gates(circuit, gate_type):
    """
    Get the list of gates of a certain type in the circuit
    
    :param circuit: CircuitDAG object
    :type circuit: CircuitDAG

    :param gate_type: string, type of the gate
    :type gate_type: str or list of str
    """

    try:
        gate_list = np.array(circuit.node_dict[gate_type])
    except:
        try:
            gate_list = np.array([])
            for i in range(len(gate_type)):
                gate_list = np.concatenate((gate_list, np.array(circuit.node_dict[gate_type[i]])))
        except:
            gate_list = np.array([])
    gate_list = np.unique(gate_list)
    return gate_list

def _Shift_Pauli_Hadamard(circuit, gate, nextgate, reg, reg_type, pauligate):

    """
    This function implement switching Pauli gates and Hadamard gates.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :param gate: The gate index of the Pauli gate
    :type gate: int

    :param nextgate: The gate index of the Hadamard gate after the Pauli gate
    :type nextgate: int

    :param reg: The register index of the emitter or photon register
    :type reg: int

    :param reg_type: the type of register for the single or double qubit gates.
    :type reg_type: "e" or "p"

    :param pauligate: The type of Pauli gate
    :type pauligate: "PauliX", "PauliY", or "PauliZ"

    :return: None
    :rtype: None
    
    """
    if pauligate == "PauliX":
        circuit.replace_op(gate, ops.Hadamard(register = reg, reg_type = reg_type))
        circuit.replace_op(nextgate, ops.SigmaZ(register = reg, reg_type = reg_type))
    elif pauligate == "PauliY":
        circuit.replace_op(gate, ops.Hadamard(register = reg, reg_type = reg_type))
        circuit.replace_op(nextgate, ops.SigmaY(register = reg, reg_type = reg_type)) #need to implement a minus sign as global phase
    elif pauligate == "PauliZ":
        circuit.replace_op(gate, ops.Hadamard(register = reg, reg_type = reg_type))
        circuit.replace_op(nextgate, ops.SigmaX(register = reg, reg_type = reg_type))

def _Shift_Pauli_Phase(circuit, gate, nextgate, reg, reg_type, pauligate):
    """
    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :param gate: The gate index of the Pauli gate
    :type gate: int

    :param nextgate: The gate index of the Phase gate after the Pauli gate
    :type nextgate: int

    :param reg: The register index of the emitter or photon register
    :type reg: int

    :param reg_type: the type of register for the single or double qubit gates.
    :type reg_type: "e" or "p"

    :param pauligate: The type of Pauli gate
    :type pauligate: "PauliX", "PauliY", or "PauliZ"

    :return: None
    :rtype: None
    
    """
    if pauligate == "PauliX":
        circuit.replace_op(gate, ops.Phase(register = reg, reg_type = reg_type))
        circuit.replace_op(nextgate, ops.SigmaY(register = reg, reg_type = reg_type))
    elif pauligate == "PauliY":
        circuit.replace_op(gate, ops.Phase(register = reg, reg_type = reg_type))
        circuit.replace_op(nextgate, ops.SigmaX(register = reg, reg_type = reg_type)) #need to implement a minus sign as a global phase
    elif pauligate == "PauliZ":
        circuit.replace_op(gate, ops.Phase(register = reg, reg_type = reg_type))
        circuit.replace_op(nextgate, ops.SigmaZ(register = reg, reg_type = reg_type))

def _get_control_target_operation(circuit, gate):
    """
    This function returns the control and target register and their types of the CNOT operation.  

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :param gate: The gate index of the CNOT gate
    :type gate: int

    :return: the control register, control register type, target register, and target register type of the CNOT gate
    :rtype: (int, str, int, str)
    
    """
    try:
        control = circuit.dag.nodes[gate]["op"].control
        control_type =circuit.dag.nodes[gate]["op"].control_type
        target = circuit.dag.nodes[gate]["op"].target
        target_type = circuit.dag.nodes[gate]["op"].target_type
        return control, control_type, target, target_type
    except:
        raise ValueError("The gate input is not a 2-qubit gate (e.g. CNOT).")
    

    

def _Shift_Pauli_CNOT(circuit, gate, pauli_reg, pauli_reg_type, pauligate, nextgate, control_reg, control_reg_type, target_reg, target_reg_type):
    """
    This function implement switching Pauli gates and CNOT gate.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :param gate: The gate index of the Pauli gate
    :type gate: int

    :param pauli_reg: The register index that has the pauli gates
    :type pauli_reg: int

    :param pauli_reg_type: the type of register has the pauli gates
    :type pauli_reg_type: "e" or "p"

    :param pauligate: The type of Pauli gate
    :type pauligate: "PauliX", "PauliY", or "PauliZ"

    :param nextgate: The gate index of the CNOT gate after the Pauli gate
    :type nextgate: int

    :param control_reg: The register index of the control qubit
    :type control_reg: int

    :param control_reg_type: the type of register for the control qubit
    :type control_reg_type: "e" or "p"

    :param target_reg: The register index of the target qubit
    :type target_reg: int

    :param target_reg_type: the type of register for the target qubit
    :type target_reg_type: "e" or "p"

    :return: None
    :rtype: None
    """
    # The following 3 lines gets the full name of the control and target register
    pauli_reg_full = pauli_reg_type + str(pauli_reg)
    control_reg_full = control_reg_type + str(control_reg)
    target_reg_full = target_reg_type + str(target_reg)

    for edge_pair in circuit.edge_dict[control_reg_type]:
        if nextgate == edge_pair[0] and control_reg_full == edge_pair[2]:
            next_next_gate_of_control = edge_pair[1]
            break

    for edge_pair in circuit.edge_dict[target_reg_type]:
        if nextgate == edge_pair[0] and target_reg_full == edge_pair[2]:
            next_next_gate_of_target = edge_pair[1]
            break

    if pauligate == "PauliX":
        if pauli_reg_full == control_reg_full:
            circuit.remove_op(gate)
            circuit.insert_at(ops.SigmaX(register = control_reg, reg_type=control_reg_type), [(nextgate, next_next_gate_of_control, control_reg_full)])
            circuit.insert_at(ops.SigmaX(register = target_reg, reg_type=target_reg_type), [(nextgate, next_next_gate_of_target, target_reg_full)])
        elif pauli_reg_full == target_reg_full:
            circuit.remove_op(gate)
            circuit.insert_at(ops.SigmaX(register = target_reg, reg_type=target_reg_type), [(nextgate, next_next_gate_of_target, target_reg_full)])
    
    elif pauligate == "PauliZ":
        if pauli_reg_full == control_reg_full:
            circuit.remove_op(gate)
            circuit.insert_at(ops.SigmaZ(register = control_reg, reg_type=control_reg_type), [(nextgate, next_next_gate_of_control, control_reg_full)])
        elif pauli_reg_full == target_reg_full:
            circuit.remove_op(gate)
            circuit.insert_at(ops.SigmaZ(register = control_reg, reg_type=control_reg_type), [(nextgate, next_next_gate_of_control, control_reg_full)])
            circuit.insert_at(ops.SigmaZ(register = target_reg, reg_type=target_reg_type), [(nextgate, next_next_gate_of_target, target_reg_full)])
    
    elif pauligate == "PauliY":
        # TODO: Implement the shifting algorithm for PauliY, first figure out the rule, then implement.
        if pauli_reg_full == control_reg_full:
            circuit.remove_op(gate)
            circuit.insert_at(ops.SigmaY(register = control_reg, reg_type=control_reg_type), [(nextgate, next_next_gate_of_control, control_reg_full)])
            circuit.insert_at(ops.SigmaX(register = target_reg, reg_type=target_reg_type), [(nextgate, next_next_gate_of_target, target_reg_full)])
        elif pauli_reg_full == target_reg_full:
            circuit.remove_op(gate)
            circuit.insert_at(ops.SigmaZ(register = control_reg, reg_type=control_reg_type), [(nextgate, next_next_gate_of_control, control_reg_full)])
            circuit.insert_at(ops.SigmaY(register = target_reg, reg_type=target_reg_type), [(nextgate, next_next_gate_of_target, target_reg_full)])

    return



def push_pauli_to_end(circuit):
    """
    This function pushes ALL the Pauli gates to the end of the circuit or before measurement.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :return: None
    :rtype: None
    
    """
    flag = True # This flag is used to check if the while loop should continue or not
    while flag:
        flag = False
        sx_list, sy_list, sz_list = _get_SigmaXYZ_list(circuit)
        for gate in sx_list:
            for edge in circuit.edge_dict["e"]:
                if gate == edge[0]:
                    try:
                        nextgate = int(edge[1])
                    except:
                        break
                    #print(type(nextgate)) # check the type of nextgate
                    reg = int(edge[2][1:]) #just to make sure that for 2 digits also works
                    #print(_get_list_of_gates(circuit, "Hadamard"))
                    if nextgate in _get_list_of_gates(circuit, "Hadamard"): 
                        _Shift_Pauli_Hadamard(circuit, gate, nextgate, reg, "e", "PauliX")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "Phase"):
                        _Shift_Pauli_Phase(circuit, gate, nextgate, reg, "e", "PauliX")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "CNOT"):
                        control, control_type, target, target_type = _get_control_target_operation(circuit, nextgate)
                        if control_type == "e" and target_type == "e":
                            _Shift_Pauli_CNOT(circuit, gate, reg, "e", "PauliX", nextgate, control, control_type, target, target_type)
                            flag = True
            '''
            for edge in circuit.edge_dict["p"]:
                if gate == edge[0]:
                    nextgate = edge[1]
                    reg = int(edge[2][1:]) #just to make sure that for 2 digits also works
                    if nextgate in _get_list_of_gates(circuit, "Hadmard"): # Need to modify if error occurs
                        _Shift_Pauli_Hadamard(circuit, gate, nextgate, reg, "p", "PauliX")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "Phase"):
                        _Shift_Pauli_Phase(circuit, gate, nextgate, reg, "p", "PauliX")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "CNOT"):
                        control, control_type, target, target_type = _get_control_target_operation(circuit, nextgate)
                        _Shift_Pauli_CNOT(circuit, gate, reg, "p", "PauliX", nextgate, control, control_type, target, target_type)
                        flag = True
            '''
        for gate in sy_list:
            for edge in circuit.edge_dict["e"]:
                if gate == edge[0]:
                    try:
                        nextgate = int(edge[1])
                    except:
                        break
                    reg = int(edge[2][1:]) #just to make sure that for 2 digits also works
                    if nextgate in _get_list_of_gates(circuit, "Hadamard"): 
                        _Shift_Pauli_Hadamard(circuit, gate, nextgate, reg, "e", "PauliY")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "Phase"):
                        _Shift_Pauli_Phase(circuit, gate, nextgate, reg, "e", "PauliY")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "CNOT"):
                        control, control_type, target, target_type = _get_control_target_operation(circuit, nextgate)
                        if control_type == "e" and target_type == "e":
                            _Shift_Pauli_CNOT(circuit, gate, reg, "e", "PauliY", nextgate, control, control_type, target, target_type)
                            flag = True
            '''
            for edge in circuit.edge_dict["p"]:
                if gate == edge[0]:
                    nextgate = edge[1]
                    reg = int(edge[2][1:]) #just to make sure that for 2 digits also works
                    if nextgate in _get_list_of_gates(circuit, "Hadmard"): # Need to modify if error occurs
                        _Shift_Pauli_Hadamard(circuit, gate, nextgate, reg, "p", "PauliY")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "Phase"):
                        _Shift_Pauli_Phase(circuit, gate, nextgate, reg, "p", "PauliY")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "CNOT"):
                        control, control_type, target, target_type = _get_control_target_operation(circuit, nextgate)
                        _Shift_Pauli_CNOT(circuit, gate, reg, "p", "PauliY", nextgate, control, control_type, target, target_type)
                        flag = True
            '''


        for gate in sz_list:
            for edge in circuit.edge_dict["e"]:
                if gate == edge[0]:
                    try:
                        nextgate = int(edge[1])
                    except:
                        break
                    reg = int(edge[2][1:]) #just to make sure that for 2 digits also works
                    if nextgate in _get_list_of_gates(circuit, "Hadamard"): 
                        _Shift_Pauli_Hadamard(circuit, gate, nextgate, reg, "e", "PauliZ")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "Phase"):
                        _Shift_Pauli_Phase(circuit, gate, nextgate, reg, "e", "PauliZ")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "CNOT"):
                        control, control_type, target, target_type = _get_control_target_operation(circuit, nextgate)
                        if control_type == "e" and target_type == "e":
                            _Shift_Pauli_CNOT(circuit, gate, reg, "e", "PauliZ", nextgate, control, control_type, target, target_type)
                            flag = True
            '''
            for edge in circuit.edge_dict["p"]:
                if gate == edge[0]:
                    nextgate = edge[1]
                    reg = int(edge[2][1:]) #just to make sure that for 2 digits also works
                    if nextgate in _get_list_of_gates(circuit, "Hadmard"): # Need to modify if error occurs
                        _Shift_Pauli_Hadamard(circuit, gate, nextgate, reg, "p", "PauliZ")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, "Phase"):
                        _Shift_Pauli_Phase(circuit, gate, nextgate, reg, "p", "PauliZ")
                        flag = True
                    elif nextgate in _get_list_of_gates(circuit, ""):
                        control, control_type, target, target_type = _get_control_target_operation(circuit, nextgate)
                        _Shift_Pauli_CNOT(circuit, gate, reg, "p", "PauliZ", nextgate, control, control_type, target, target_type)
                        flag = True
                    '''
    return 


def _get_index_from_full_name(full_name):
    """
    This function returns the index of the register from the full name of the register.

    :param full_name: The full name of the register
    :type full_name: str

    :return: the index of the register
    :rtype: int
    
    """
    return int(full_name[1:])

def _get_emitter_clifford_block(original_circuit):
    """
    This gets the first emitter clifford block that we want to optimize, and the sub-circuit of the rest of the circuit.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :return: the subcircuit of the emitter clifford block that we want to optimize and the rest of the circuit.
    :rtype: 2 tuple of CircuitDAG object
    
    """

    circuit = original_circuit.copy()
    n_emitter = circuit.n_emitters

    gates_before_emission = [] 
    """    for i in range(n_emitter):
        gates_before_emission.append([])"""
    first_e_p_gate_emitter = np.ones(n_emitter)
    first_e_p_gate_emitter *= -1
    CNOT_list = _get_list_of_gates(circuit, "CNOT")
    CNOT_between_e_p = []
    Classicaltwoqubit_list = _get_list_of_gates(circuit, ["ClassicalCNOT", "MeasurementCNOTandReset"])
    for gate in CNOT_list:
        control, control_type, target, target_type =_get_control_target_operation(circuit, gate)
        if control_type == "e" and target_type == "p":
            CNOT_between_e_p.append(gate)
    CNOT_between_e_p = np.array(CNOT_between_e_p)
    emitter_non_unitary = np.concatenate((CNOT_between_e_p, Classicaltwoqubit_list))

    Emitter_emitter_CNOT = _get_list_of_gates(circuit, "Emitter-Emitter")
    try:
        Emitter_emitter_CNOT_check = np.zeros(max(Emitter_emitter_CNOT)+1)
    except:
        Emitter_emitter_CNOT_check = np.zeros(1)


    while -1 in first_e_p_gate_emitter:
        for i in range(n_emitter):
            if first_e_p_gate_emitter[i] != -1:
                continue
            emit_name = "e" + str(i)
            previous_gate = emit_name + "_in"
            flag = True
            first_time = True
            while flag:
                for edge in circuit.edge_dict["e"]:
                    if edge[2] == emit_name and edge[0] == previous_gate:
                        try: 
                            this_gate = int(edge[1])
                        except:
                            if edge[1] == "e" + str(i) + "_out":
                                flag = False
                                first_e_p_gate_emitter[i] = -2
                                break
                            elif edge[0] == "e" + str(i) + "_in":
                                continue
                        previous_gate = edge[1]
                        if this_gate in gates_before_emission or this_gate in first_e_p_gate_emitter:
                            continue
                        if this_gate in emitter_non_unitary:
                            if first_time:
                                first_e_p_gate_emitter[i] = this_gate
                                first_time = False
                            flag = False
                            break
                        elif this_gate in Emitter_emitter_CNOT:
                            control, control_type, target, target_type =_get_control_target_operation(circuit, this_gate)
                            control_name = control_type + str(control)
                            target_name = target_type + str(target)
                            if emit_name == control_name:
                                if Emitter_emitter_CNOT_check[this_gate] == 1:
                                    if first_e_p_gate_emitter[target] != -1:
                                        first_e_p_gate_emitter[i] = -2
                                        flag = False
                                        break
                                    elif first_e_p_gate_emitter[target] == -1:
                                        flag = False
                                        break
                                Emitter_emitter_CNOT_check[this_gate] += 1
                                if Emitter_emitter_CNOT_check[this_gate] == 0:
                                    gates_before_emission.append(this_gate)

                                elif Emitter_emitter_CNOT_check[this_gate] == 1:
                                    flag = False
                                    break
                                
                            elif emit_name == target_name:
                                if Emitter_emitter_CNOT_check[this_gate] == -1:
                                    if first_e_p_gate_emitter[control] != -1:
                                        first_e_p_gate_emitter[i] = -2

                                        flag = False
                                        break
                                    elif first_e_p_gate_emitter[target] == -1:
                                        flag = False
                                        break
                                Emitter_emitter_CNOT_check[this_gate] -= 1
                                if Emitter_emitter_CNOT_check[this_gate] == 0:
                                    gates_before_emission.append(this_gate)

                                elif Emitter_emitter_CNOT_check[this_gate] == -1:
                                    flag = False
                                    break
                        else:
                            gates_before_emission.append(this_gate)

    subcircuit = original_circuit.copy()

    all_gates = _get_list_of_gates(original_circuit, ["one-qubit", "two-qubit"])

    gates_before_emission = np.array(gates_before_emission)
    gates_before_emission = np.unique(gates_before_emission)
    for gate in all_gates:
        if gate not in gates_before_emission:
            subcircuit.remove_op(gate)
    
    for gate in gates_before_emission:
        circuit.remove_op(gate)
    for gate in first_e_p_gate_emitter:
        if gate > 0 :
            circuit.remove_op(gate)


    return subcircuit, circuit
                    
def _get_all_emitter_clifford_blocks(original_circuit):
    """
    This gets all the emitter clifford blocks that we want to optimize, and the sub-circuit of the rest of the circuit.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :return: the subcircuit of the emitter clifford block that we want to optimize and the rest of the circuit.
    :rtype: list of CircuitDAG object
    """

    circuit = original_circuit.copy()
    all_emitter_clifford_block = []
    first_emitter_clifford, rest_of_circuit = _get_emitter_clifford_block(circuit)
    while len(_get_list_of_gates(rest_of_circuit, ["Emitter", "Emitter-Emitter"])) > 0:
        all_emitter_clifford_block.append(first_emitter_clifford)
        first_emitter_clifford, rest_of_circuit = _get_emitter_clifford_block(rest_of_circuit.copy())
    all_emitter_clifford_block.append(first_emitter_clifford)
    return all_emitter_clifford_block

def _Number_of_emitter_CNOT(original_circuit):
    """
    This function returns the number of emitter-emitter CNOT gates in the emitter clifford block.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :return: the number of emitter-emitter CNOT gates in the emitter clifford block.
    :rtype: int
    """

    original_CNOT = _get_list_of_gates(original_circuit, "CNOT")
    original_emitter_photonic = _get_list_of_gates(original_circuit, "Emitter-Photonic")
    original_CNOT_number = len(original_CNOT)
    for i in original_CNOT:
        if i in original_emitter_photonic:
            original_CNOT_number -= 1
    return original_CNOT_number


def block_optimize_with_qiskit_CNOTcount(circuit, n):
    """
    This function optimizes the CNOT count clifford block with qiskit.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :param n: The number of qubits in the circuit
    :type n: int

    :return: None
    :rtype: None
    """
    import qiskit as qisk
    from qiskit import QuantumCircuit, transpile
    import qiskit.qasm2 as qq2
    from qiskit.providers.fake_provider import GenericBackendV2
    import matplotlib.pyplot as plt

    original_qasm = circuit.to_openqasm()

    qiskitcircuit = qq2.loads(original_qasm)
    backend = GenericBackendV2(n)
    overall_optimized_CNOTcount = 999
    for level in range(4):
        circ = transpile(qiskitcircuit, backend, optimization_level=level)
        try:
            optimized_CNOTcount = circ.count_ops()["cx"]
        except:
            optimized_CNOTcount = 0
        if optimized_CNOTcount < overall_optimized_CNOTcount:
            overall_optimized_CNOTcount = optimized_CNOTcount
    return overall_optimized_CNOTcount


def block_optimize_with_pytket_CNOTcount(circuit, n):
    """
    This function returns the CNOT count of circuit before and after optimization.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :param n: The number of qubits in the circuit
    :type n: int

    :return: optimized CNOT count
    :rtype: int
    
    """
    from qiskit import transpile
    import pytket as pk
    import pytket.passes as ps
    import pytket.qasm as pq
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    import qiskit.qasm2 as qq2

    original_qasm = circuit.to_openqasm()
    qiskitcircuit = qq2.loads(original_qasm)
    qiskitcircuit = qiskitcircuit.decompose()
    circuit = qiskit_to_tk(qiskitcircuit)
    ps.CliffordSimp(allow_swaps=False).apply(circuit)
    qiskitcircuit = tk_to_qiskit(circuit)
    overall_optimized_CNOTcount = 0
    try:
        overall_optimized_CNOTcount = qiskitcircuit.count_ops()["cx"]
    except:
        overall_optimized_CNOTcount = 0
    return overall_optimized_CNOTcount



def Number_of_CNOT_before_after_optimized_with_qiskit(original_circuit):

    """
    This function returns the number of emitter-emitter CNOT gates before and after the optimization.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :return: the number of emitter-emitter CNOT gates before and after the optimization.
    :rtype: 2 tuple of int   
    """
    original_CNOT_number = _Number_of_emitter_CNOT(original_circuit)
    all_emitter_clifford_block = _get_all_emitter_clifford_blocks(original_circuit)
    optimized_CNOT_number = 0
    n_emitter = original_circuit.n_emitters
    n_photon = original_circuit.n_photons
    for circuit in all_emitter_clifford_block:
        optimized_CNOT_number += block_optimize_with_qiskit_CNOTcount(circuit, n_emitter + n_photon)

    return original_CNOT_number, optimized_CNOT_number


def Number_of_CNOT_before_after_optimized_with_pytket(original_circuit):

    """
    This function returns the number of emitter-emitter CNOT gates before and after the optimization.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :return: the number of emitter-emitter CNOT gates before and after the optimization.
    :rtype: 2 tuple of int   
    """
    original_CNOT_number = _Number_of_emitter_CNOT(original_circuit)
    all_emitter_clifford_block = _get_all_emitter_clifford_blocks(original_circuit)
    optimized_CNOT_number = 0
    n_emitter = original_circuit.n_emitters
    n_photon = original_circuit.n_photons
    for circuit in all_emitter_clifford_block:
        optimized_CNOT_number += block_optimize_with_pytket_CNOTcount(circuit, n_emitter + n_photon)

    return original_CNOT_number, optimized_CNOT_number












#--------------------------------------------------------------------------------------------
"""
The following functions are still under constrcution.
They are used to find the optimal cut for the emitter clifford block and optimize.
"""

def _all_possible_list_index(numlist):

    """
    This function returns all the possible index of the list.

    :param alist: The list that we want to get all the possible index.
    :type alist: list

    :return: all the possible index of the list.
    :rtype: list of list of int
    """
    if len(numlist) == 1:
        all_possible_list_index = []
        for i in range(numlist[0]):
            all_possible_list_index.append([i])
        return all_possible_list_index

    else:
        all_possible_list_index = []
        for i in range(numlist[0]):
            for j in _all_possible_list_index(numlist[1:]):
                all_possible_list_index.append([i] + j)
        return all_possible_list_index

def _all_possible_list_index_with_1_maxi(numlist):
    all_possible_list_index_with_1_maxi = []
    for i in range(len(numlist)):
        all_possible_list_index = _all_possible_list_index(numlist[0:i]+numlist[i+1:])
        for list_of_index in all_possible_list_index:
            list_of_index.insert(i, numlist[i]-1)
            all_possible_list_index_with_1_maxi.append(list_of_index)
    return all_possible_list_index

def _keep_only_gates(original_circuit, list_of_gates, list_of_e_p_emission):

    """
    This function keeps only the gates in the list_of_gates in the circuit.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :param list_of_gates: The list of gates that we want to keep in the circuit.
    :type list_of_gates: list of gate indices

    :return: two circuits, the first one contains only gates, the second one contains the rest of the circuit.
    :rtype: 2 tuple of CircuitDAG object
    """
    all_gates = _get_list_of_gates(original_circuit, ["one-qubit", "two-qubit"])
    subcircuit = original_circuit.copy()
    rest_circuit = original_circuit.copy()
    for gate in all_gates:
        if gate not in list_of_gates:
            subcircuit.remove_op(gate)
    for gate in list_of_gates:
        rest_circuit.remove_op(gate)
    return subcircuit, rest_circuit

def _all_possible_list_index(numlist):

    """
    This function returns all the possible index of the list.

    :param alist: The list that we want to get all the possible index.
    :type alist: list

    :return: all the possible index of the list.
    :rtype: list of list of int
    """
    if len(numlist) == 1:
        all_possible_list_index = []
        for i in range(numlist[0]):
            all_possible_list_index.append([i])
        return all_possible_list_index

    else:
        all_possible_list_index = []
        for i in range(numlist[0]):
            for j in _all_possible_list_index(numlist[1:]):
                all_possible_list_index.append([i] + j)
        return all_possible_list_index
    

def _all_possible_list_index_with_1_maxi(numlist):
    all_possible_list_index_with_1_maxi = []
    for i in range(len(numlist)):
        all_possible_list_index = _all_possible_list_index(numlist[0:i]+numlist[i+1:])
        for list_of_index in all_possible_list_index:
            list_of_index.insert(i, numlist[i]-1)
            all_possible_list_index_with_1_maxi.append(list_of_index)
    return all_possible_list_index_with_1_maxi
    
def _get_maxi_index(numlist, all_possible_list_index):
    maxi_index = []
    for list_index in all_possible_list_index:
        this_maxi_index = []
        for i in range(len(numlist)):
            if numlist[i] - 1 == list_index[i]:
                this_maxi_index.append(i)
        maxi_index.append(this_maxi_index)
    return maxi_index



def _find_first_emitter_all_possible_clifford(original_circuit):
    """
    This function returns all the possible sets of clifford gates in the circuit.

    :param circuit: The circuit object that contains the circuit information.
    :type circuit: CircuitDAG object

    :return: all the possible sets of clifford gates in the circuit.
    :rtype: list of list of int
    """
    circuit = original_circuit.copy()
    n_emitter = circuit.n_emitters
    #n_photon = circuit.n_photons
    #n_classical = circuit.n_classical

    gates_after_emission = [] 
    first_e_p_gate_emitter = np.zeros(n_emitter)

    # The following code is to find the first emitter gate before the photon emission or measurement
    CNOT_list = _get_list_of_gates(circuit, "CNOT")
    CNOT_between_e_p = []
    Classicaltwoqubit_list = _get_list_of_gates(circuit, ["ClassicalCNOT", "MeasurementCNOTandReset"])
    for gate in CNOT_list:
        control, control_type, target, target_type =_get_control_target_operation(circuit, gate)
        if control_type == "e" and target_type == "p":
            CNOT_between_e_p.append(gate)
    CNOT_between_e_p = np.array(CNOT_between_e_p)
    emitter_non_unitary = np.concatenate((CNOT_between_e_p, Classicaltwoqubit_list))


    # now I try to find the first emitter gate before the photon emission or measurement
    gates_before_first_emission = []
    all_possible_sets = []
    for i in range(n_emitter):
        gates_before_first_emission.append([])
        emit_name = "e" + str(i)
        previous_gate = emit_name + "_in"
        flag = True
        #signal = False
        #first_time = True
        while flag:
            for edge in circuit.edge_dict["e"]:
                if edge[2] == emit_name and edge[0] == previous_gate:
                    previous_gate = edge[1]
                    try: 
                        secondgate = int(edge[1])
                        gates_before_first_emission[i].append(secondgate)
                    except:
                        if edge[1] == "e" + str(i) + "_out":
                            flag = False
                            break
                        elif edge[0] == "e" + str(i) + "_in":
                            continue
                        break
                    if secondgate in emitter_non_unitary:
                        flag = False
                        break
    
    
    all_possible_list_index_with_1_maxi = _all_possible_list_index_with_1_maxi([len(gates_before_first_emission[i]) for i in range(n_emitter)])
    for list_of_index in all_possible_list_index_with_1_maxi:
        this_circuit_gates = gates_before_first_emission
        for i in range(n_emitter):
            this_circuit_gates[i] = gates_before_first_emission[i][0:list_of_index[i]]
        this_clifford_block_circuit = original_circuit.copy()    
        this_rest_of_circuit = original_circuit.copy()
        all_possible_list_index = _all_possible_list_index([len(gates_before_first_emission[i]) for i in range(n_emitter)])
        for list_index in all_possible_list_index:
            first_clifford_gates = gates_before_first_emission.copy()
            for i in range(len(list_index)):
                first_clifford_gates[i] = first_clifford_gates[i][0:list_index[i]]
        subcircuit, rest_circuit = _keep_only_gates(circuit, first_clifford_gates)
        while len(_get_list_of_gates(rest_of_circuit, "Emitter")) > 0:
            # have bug, need to fix that e0_out issue.
            all_emitter_clifford_block.append(first_emitter_clifford)
            first_emitter_clifford, rest_of_circuit = _get_emitter_clifford_block(rest_of_circuit.copy())
        # try to use recursion to get the rest of the clifford gates
        #try to implement the rest of the function by keeping only the gates


                