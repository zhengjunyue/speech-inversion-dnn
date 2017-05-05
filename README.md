# speech-inversion-dnn
Acoustic to Articulatory speech inversion with a Feedforward DNN

Code to estimate Vocal Tract constriction Variables (TVs) from speech using a Deep Neural Network trained with Keras in Python. The Network is trained to estimate 6 Tract Variables (LA, LP, TBCL, TBCD, TTCL, TTCD) from contextualized MFCCs

Usage: python estimate_TV_xrmb.py <path/to/audio/file> <output/directory>

Output is saved as a .mat file in the Matlab format. Feature dimension per frame = 6.
