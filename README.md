# speech-inversion-dnn
Acoustic to Articulatory speech inversion with a Feedforward DNN

Code to estimate Vocal Tract constriction Variables (TVs) from speech using a Deep Neural Network trained with Keras in Python. The Network is trained to estimate 6 Tract Variables (LA, LP, TBCL, TBCD, TTCL, TTCD) from contextualized MFCCs

Usage: python estimate_TV_xrmb.py <path/to/audio/file> <output/directory>

Output is saved as a .mat file in the Matlab format. Feature dimension per frame = 6.


# If you found this code useful for your research, please cite the following paper and thesis:

Sivaraman, Ganesh, Vikramjit Mitra, Hosung Nam, Mark Tiede, and Carol Espy-Wilson. 2016. “Vocal Tract Length Normalization for Speaker Independent Acoustic-to-Articulatory Speech Inversion.” Pp. 455–59 in Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, vol. 08–12–Sept. 

Bibtex entry:
@inproceedings{Sivaraman2016a,
author = {Sivaraman, G. and Mitra, V. and Nam, H. and Tiede, M. and Espy-Wilson, C.},
booktitle = {Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH},
doi = {10.21437/Interspeech.2016-1399},
issn = {19909772},
keywords = {Acoustic to articulatory speech inversion,Speaker normalization,Vocal Tract Length Normalization},
title = {{Vocal tract length normalization for speaker independent acoustic-to-articulatory speech inversion}},
year = {2016}
}

Sivaraman, Ganesh. 2017. “Articulatory Representations to Address Acoustic Variability in Speech.”, PhD thesis, University of Maryland College Park.
Bibtex entry:
@phdthesis{Sivaraman2017a,
author = {Sivaraman, Ganesh},
school = {University of Maryland College Park},
title = {{Articulatory representations to address acoustic variability in speech}},
year = {2017}
}

