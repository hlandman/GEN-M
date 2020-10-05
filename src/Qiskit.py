# import qiskit
from qiskit.tools.visualization import circuit_drawer

from qiskit import ClassicalRegister, QuantumRegister  # , QuantumProgram
from qiskit import QuantumCircuit, execute

from qiskit import Aer, IBMQ

import struct
import numpy as np
import math
from scipy import signal as sg
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def genWave(f=440, T=1, phase=0, type='sine'):
    Fs = 44100  ## Sampling Rate
    ## Frequency (in Hz)
    sample = 44100 * T  ## Number of samples
    x = np.arange(sample)

    if type == 'sine':
        ####### sine wave ###########
        y = np.sin((2 * np.pi * f * x + phase) / Fs)
    elif type == 'square':
        ####### square wave ##########
        y = sg.square((2 * np.pi * f * x + phase) / Fs)
    elif type == 'duty':
        ####### square wave with Duty Cycle ##########
        y = sg.square((2 * np.pi * f * x + phase) / Fs, duty=0.8)
    elif type == 'saw':
        ####### Sawtooth wave ########
        y = sg.sawtooth((2 * np.pi * f * x + phase) / Fs)

    return y


def sineadd(s1, s2, a1, a2):
    l1 = len(s1)
    l2 = len(s2)
    s1 = s1[0:min(l1, l2)]
    s2 = s2[0:min(l1, l2)]

    added = (np.add(a1 * s1, a2 * s2))
    return added  # /np.max(added)


def bloch_sonify(phi_theta, filename, dur=0.05, repeat=0):
    phase_weight = 20
    F0 = 262
    F1 = 330
    stotal = np.array([])
    for angles in phi_theta:
        # print(angles)
        s1 = genWave(F0 - angles[0] * phase_weight, dur, 0, 'sine')
        # s2 = genWave(F1+angles[0],0.25,angles[0],'saw')
        s2 = genWave(F1 + angles[0] * phase_weight, dur, 0, 'saw')
        s3 = sineadd(s1, s2, math.pi - angles[1], angles[1])
        stotal = np.concatenate((stotal, s3))
    # stotal = savgol_filter(stotal, 51, 3)
    # stotal = savgol_filter(stotal, 251, 3)
    if np.max(np.abs(stotal), axis=0) != 0:
        stotal /= np.max(np.abs(stotal), axis=0)
    for i in range(0, repeat):
        stotal = np.concatenate((stotal, stotal))
    stotal = savgol_filter(stotal, 251, 3)
    wavfile.write(filename, 44100, stotal)


def f_midi(midi):
    f = 55 * (pow(2, (midi - 33) / 12))
    return f


# assumes 3 qubits
def results_sonify(results, T=2, octave=2, wavetype='sine', repeat=0, compress=True, Fseq=0):
    # results is a list of lists
    # c3,e3,g3, b3,d4,f4,a4,c5

    Fmap_midi = [[48, 52, 55, 59, 62, 65, 69, 72], [55, 59, 62, 65, 67, 71, 74, 77], [53, 57, 60, 65, 69, 72, 77, 81]]
    Fmap = []
    for i in Fmap_midi:
        Fmap_f = []
        for j in i:
            Fmap_f.append(f_midi(j))
        Fmap.append(Fmap_f)
    Fmap = list(Fmap[Fseq])
    Fmap_old = [130.81, 164.81, 196, 246.94, 293.67, 349.23, 440, 523.25]
    print(Fmap_old)
    print(Fmap)

    N = len(results)

    stotal = np.array([])
    # each r will be 3 qubits - i.e. 8 counts
    # each of the 8 counts is then mapped onto the amplitude of Fmap sines
    for r in results:
        sines = r[0] * genWave(octave * Fmap[0], T, 0, wavetype)
        # r /= np.max(np.abs(r),axis=0)
        for count_index in range(1, len(r) - 1):
            if compress:
                sines += math.sqrt(r[count_index]) * genWave(octave * Fmap[count_index], T, 0, wavetype)
        # normalise
        if np.max(np.abs(sines), axis=0) != 0:
            sines /= np.max(np.abs(sines), axis=0)
        # so now we have a chord for one set of results
        stotal = np.concatenate((stotal, sines))
    for i in range(0, repeat):
        stotal = np.concatenate((stotal, stotal))
    stotal = savgol_filter(stotal, 151, 3)
    # stotal = savgol_filter(stotal, 651,3)
    return stotal


# results = [[100,150,50,36,85,27,85,98],[29,48,238,48,29,59,48,290],[38,57,30,28,568,38,38,49,30]]
# x = results_sonify(results,wavetype="saw")
# wavfile.write("results_son.wav",44100,x)


# qasm_root = 'C:\\Users\\ajkirke\\Dropbox\\Work\\python\\IBMMusic\\QISKitTests\\'
# qasm_root = 'C:/Users/ajkirke/Dropbox/Work/python/
backend = Aer.get_backend('qasm_simulator')  # this is the backend that will be used

q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q, c)


# qasm_file= "alexis_qasm.qasm"
# qasm_file =qasm_root + qasm_file
# qasm_file= "alexis_Grover.qasm"

def run_qasm(qasm_list):
    with open('temp_qasm_file.qasm', 'w') as f:
        for item in qasm_list:
            f.write("%s\n" % item)
    qCircuit = QuantumCircuit.from_qasm_file('temp_qasm_file.qasm')  # (qasm_file)
    job_sim = execute(qCircuit, backend, shots=1024)
    count_dict = job_sim.result().get_counts(qCircuit)
    print(count_dict)
    result_items = ['000', '001', '010', '011', '100', '101', '110', '111']
    # shows which of result_items is in count_dict
    in_results = list(count_dict)

    # This will turn it into results with 0s inserted for those not returned
    full_results = []
    for ri in result_items:
        if ri in in_results:
            full_results.append(count_dict[ri])
        else:
            full_results.append(0)
    return full_results


def sonify_qasm(qasm_file, write=True, wavetype="sine", Fseq=[0], T=1, repeat=0, octave=2, filename=""):
    if filename == "":
        filename = "sonified_" + qasm_file + ".wav"
    with open(qasm_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    print(content)

    header = []
    group = []
    groups = []
    footer = []
    for line in content:
        if line != '':
            group.append(line)
            # print('appending ', line)
        else:
            if 'include' in group[0]:
                header = group
                group = []
                # print('reset after header')
            elif 'measure' in group[0]:
                footer = group
                group = []
                # print('reset after footer')
            else:
                groups.append(group)
                group = []
                # print('reset after group')
    if footer == []:
        footer = group

    # print('header ', header)
    # print('groups ', groups)
    # print('footer ', footer)

    independent = False
    results = []
    prepper = []
    g_cum = []
    groups = [''] + groups
    for g in groups:

        if independent:
            code = header + g + footer
            # print(code)
            results.append(run_qasm(code))
        else:
            g_cum += g
            code = header + g_cum + footer
            print("cum_code ", code)
            results.append(run_qasm(code))

    # print(results)
    if write:
        x = results_sonify(results, wavetype=wavetype, T=T, Fseq=Fseq, repeat=repeat, octave=octave)
        wavfile.write(filename, 44100, x)
        print("******WAV written*****")

    return (results)


phi = 0
phi_theta = []
while phi < 2 * math.pi:
    theta = 0
    while theta < math.pi:
        # print(round(phi,2),round(theta,2))
        phi_theta.append([phi, theta])
        theta += 0.3
    phi += 0.3

phi_theta_circuit = [[1.3, 1.5], [2.0853981633974485, 1.5], [-2.6269908169872416, 1.5],
                     [2.6269908169872416, 1.6415926535897933], [0.5146018366025518, 1.5]]

# results = sonify_qasm('alexis_Grover.qasm', filename = "grover2.wav",Fseq =0, T= 6, octave = 1, repeat = 4)
results2 = sonify_qasm(qasm_file='alexis_teleport.qasm', filename="teleport.wav", Fseq=2, T=3, octave=2, repeat=4)
bloch_sonify(phi_theta_circuit, "bloch_son.wav", 0.5, repeat=4)