#data-raw file

#load 
library(oro.dicom)
test_dcm = file.path(.filepaths$train_dcim, 'ISIC_0078703.dcm')
#test_dcm <- file.path('/home/udaman/Documents/ISIC_0015719.dcm')
dcm <- readDICOMFile(test_dcm, verbose = TRUE)
