# NeuralWaveToF
Finding optimal coding functions for continuous wave time of flight with neural networks  


## Table of Contents
1. [ML Environment Setup](#mlsetup)

<a name="mlsetup">
</a>

## Tensorflow and Keras Setup 

We will use anaconda to create a virtual environment that uses python 2.7. Why 2.7 and not 3.5? I saw a few github issues and posts online talking about problems running Keras with Theano on python 3.5.

1. Setup anaconda
2. Create an conda environment with the dependencies:

```
    conda create --name mlenv python=2.7 numpy scipy pandas scikit-learn h5py
```

3. Activate environment

```
    source activate mlenv
```

4. Install tensorflow (for cpu only):

```
    conda install -c conda-forge tensorflow
```


5. Install keras dependencies:

```
    conda install yaml
```

6. Install keras (make sure 2.0 is installed because it is suppose to be beter integrated with TF):

```
    conda install -c conda-forge keras=2.0.2
```

**NOTE 1:** Make sure that the previous command will install tensorflow for python 2.7!

**NOTE 2:** Make sure that all commands after 3 ran with `mlenv` activated.

TODOS:
1. Check if keras was setup with a tensorflow or theano backend

2. Figure out how to configure cuDNN for GPU acceleration. 
