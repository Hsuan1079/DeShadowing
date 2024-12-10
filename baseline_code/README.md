# baseline brightness 
* The code intuitively leverages Otsu's automatic thresholding to segment brightness and identify shadow regions. It then applies smooth mask edges, enhances brightness in the shadowed areas.
# baseline histogram
* The code intuitively leverages Otsu's automatic thresholding to segment brightness and identify shadow regions. It then merges smaller shadow segments and restores shadowed areas through histogram matching.
# matlab pairing
* A baseline that also uses pairing for shadow removal
* The reference code is from https://github.com/kittenish/Image-Shadow-Detection-and-Removal/tree/master
* The code only modifies the input and output, and makes up for the missing parts of the original program code meanshift that make it unexecutable.
