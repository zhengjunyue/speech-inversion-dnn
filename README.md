# speech-inversion-matlabNN
Code to estimate Vocal Tract constriction Variables (TVs) from speech using a 3 layer deep Neural Network trained in Python with Keras
The Network is trained to estimate 6 Tract Variables (LA, LP, TBCL, TBCD, TTCL, TTCD) from contextualized MFCCs

Usage:
python estimate_TV_xrmb.py \<path/to/audio/file> \<output/directory>

Output is saved as a .htk file in the HTK feature format. Feature dimension per frame = 6.

Possible errors/issues:

Thre is a possibility that the HCopy.exe attached here does not match the OS you are running on
To fix this, download HTK from http://htk.eng.cam.ac.uk/download.shtml
and install it. Make sure to add the compiled binaries of HTK to your path
