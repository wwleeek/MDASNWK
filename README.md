# MDASNWK
The method of disease-metabolite prediction association.
## Data
Used to test the model(MDASNWK)
### D-M_337x1444.txt
known disease-metabolite associations
### DJS.txt
Integrated disease-disease similarity
### MJS.txt
Integrated metabolite-metabolite similarity
## Model
This file is the model of disease-metabolite prediction association
### MDASNWK
Model implementation file
### cosSim dist_E selftuning2 
Similarity calculation method
### MDASNWK5KCV
This file is the main function of the method(MDASNWK)in 5-fold CV
### MDASNWKLOOCV
This file is the main function of the method(MDASNWK)in LOOCV
## NPU
This file is an example of the model implemented on the NPU
### application
This file is the operator model application execution file
### constant
This file is the parameter definition file
### NPU operator development process
https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/operatordev/Ascendcopdevg/atlas_ascendc__10_0004.html
### NPU application development process
https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/inferapplicationdev/aclpythondevg/aclpythondevg_0000.html

