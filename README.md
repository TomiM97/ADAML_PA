# {Predicting leaf and canopy traits using hyperspectral data}

 
  

##  Overview

Creating	individual prediction models for leaf and canopy traits. Predicting the traits using PLS regression model. After having a complete trait matrix, applying PCA on the trait matrix and interpreting variational profiles of the traits.		

## Expolatory analysis

Simple analysis on the data was done using PCA and other methods. Code needed can be found in [PW](https://github.com/TomiM97/ADAML_PA/blob/master/PW_PCA.m).

## Initial model
Initial models were created using PLS1. The code for this is in [PLS1_R2](https://github.com/TomiM97/ADAML_PA/blob/master/PLS1_R2.m). It uses the functions [pls1_R2_cal](https://github.com/TomiM97/ADAML_PA/blob/master/pls1_R2_cal.m) and [r2_pls1_forward_selection](https://github.com/TomiM97/ADAML_PA/blob/master/r2_pls1_forward_selection.m).

Another PLS1 model is in [PW_CV_PLS](https://github.com/TomiM97/ADAML_PA/blob/master/PW_CV_PLS.m). The [extented multiscatter correction function](https://github.com/TomiM97/ADAML_PA/blob/master/emsc_fun.m) and [Gaussian process function](https://github.com/TomiM97/ADAML_PA/blob/master/GP_fun_t.m) were used in this part.

## Kernel-PLS
K-PLS code is in [PW_CV_KPLS](https://github.com/TomiM97/ADAML_PA/blob/master/PW_CV_KPLS.m). Functions for that are [createDataset](https://github.com/TomiM97/ADAML_PA/blob/master/createDataset.m),[datasetSplit](https://github.com/TomiM97/ADAML_PA/blob/master/datasetSplit.m),[fit](https://github.com/TomiM97/ADAML_PA/blob/master/fit.m),[kernelRBF](https://github.com/TomiM97/ADAML_PA/blob/master/kernelRBF.m),[kernelRBFTest](https://github.com/TomiM97/ADAML_PA/blob/master/kernelRBFTest.m),[kernel_predict](https://github.com/TomiM97/ADAML_PA/blob/master/kernel_predict.m),[kpls](https://github.com/TomiM97/ADAML_PA/blob/master/kpls.m),[normDiff](https://github.com/TomiM97/ADAML_PA/blob/master/normDiff.m),[predict](https://github.com/TomiM97/ADAML_PA/blob/master/predict.m),[kerneö_R2_cal](https://github.com/TomiM97/ADAML_PA/blob/master/kernel_R2_cal.m).

