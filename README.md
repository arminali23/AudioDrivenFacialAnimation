# AudioDrivenFacialAnimation

- Project includes facial animation by audio using deep learning model and VOCA dataset 

- Project folder has initial .py files version of the main pipeline also with jupyter notebook version where the visualizations are
- Project also has two different jupyter notebook files which includes;
  - Experiments with processed data
  - Experiments without pca normalization using init_basis_expression.npy
- These two approaches failed to have a predicition with mouth movement, where there was a movement in predicition in the notebook without normalization but, it has been clarified that it is a noise that looks like a movement. The noise comes from init_basis_expression

- with pca normalization and excluding init_basis_expression file, project satisfied with a decent prediction.

- referencing to the dataset

- @inproceedings{VOCA2019,
    title = {Capture, Learning, and Synthesis of {3D} Speaking Styles},
    author = {Cudeiro, Daniel and Bolkart, Timo and Laidlaw, Cassidy and Ranjan, Anurag and Black, Michael},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    pages = {10101--10111},
    year = {2019}
    url = {http://voca.is.tue.mpg.de/}
}
