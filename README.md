# DECODED
## What is BMIsLab?
The Brain-Machine Interface Systems Lab (BMISLAB) is a team of researchers led by José María Azorín. Our work focuses on human-machine interaction through brain control in order to improve human capabilities in neural rehabilitation. This repository is an open source signal processing environment for electroencephalography (EEG) signals running on Matlab. This folder contains original Matlab functions from the BMISLAB that have been adapted to the context of the EUROBENCH FTSP-2 subproject, DECODED.

## Sub-directories
- /ReadFiles - Folder from which the EEG data will be read to compute the motor imagery and attention index
- /Results - Folder from which the index and all the results generated will be stored

## To use DECODED code:
1. Start Matlab
2. Use Matlab to navigte to the folder containing the code
3. Change variables as needed:
  - plot_results : to indicate if you want to generate a plot with the results
  - shift : the number of samples of every EEG epoch
  - n_electrodes = number of electrodes in the data
  - fm : sampling frequency
  - electrodes_selected : name of the electrodes to be used during classification
  - processing_windows : number of samples of every processing window
4. Execute DECODED_PIs
5. At the end of the program, user must input the name of the folder to save the results. A new folder with this name will be created in the "/Results" folder
