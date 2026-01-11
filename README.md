## January Updates

It was a fun holiday. First IPs have been filed. Several POCs done with all-optical inference to explore the envelope of what is possible:
```
AlphaGo, a unitary ish Born-rule collapse all-optical Go player (this is from one overnight training)
https://github.com/dwallener/EntropicaPublic/blob/main/v0.4-alphago/README.md
```

## Background (WIP)

We present Entropica, the first generative language model whose forward pass is physically realizable as a passive linear-optical interferometer operating at zero electrical power during inference. 

The model uses a 1024-dimensional complex Hilbert space with 32 layers of programmable Mach–Zehnder meshes (Reck architecture) and derives token probabilities directly via the Born rule on a 650 nm laser beam. 

Despite using only unitary operations and no attention mechanism, a 1024×32 model achieves coherent TinyStories generation after < 1.8 hours of training on a single consumer GPU. We further demonstrate a complete optical implementation path using printed phase masks on transparency film and a $30 laser diode. 


Most recent technical note: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17764289.svg)](https://doi.org/10.5281/zenodo.17764289)

https://zenodo.org/records/17764289 


![image (2)](https://github.com/user-attachments/assets/8f452693-2b84-47f0-8a22-0a5bf68b19ce)

## Public Facing Repo for Quantum-LM Exploration

### v001

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17764289.svg)](https://doi.org/10.5281/zenodo.17764289)

https://zenodo.org/records/17764289 

The source code for the first Entropica paper. The model itself is in v000/quantum_lm.py

https://github.com/dwallener/EntropicaPublic/blob/main/v001/quantum_lm.py 

The scripts generate/sample/train manage the dataset, inference and training process.
