# Time-Viz

This is the repository for the "Visualising deep network representations of Limit Order Book time-series data" 
by Błażej Leporowski and Alexandros Iosifids.

The code has been written by Błażej Leporowski, with parts of it taken from the publicly available implementation of the 
parametric t-SNE classifier by Jacob Slittera found at https://github.com/jsilter/parametric_tsne, and implementation of 
the TABL classifier by Dat Tran found at https://github.com/viebboy/TABL.


Instructions:
<ol>
<li>Create a conda environment using 'conda create --name <env> --file <this file>', or equivalent commands for other 
enviroments.</this>
<li>Open file Source/meta.py</li>
<li>Fill in the correct paths for loading the data, saving the result and calculations for further use.</li>
<li>Fill in the experiment parameters, or leave the default. By default the program will create a finetuned Time-Viz 
model based on TABL classifier, that is the best performing one from the paper.</li>
<li>Results will be visible in folder Results</li>
</ol>	

Due to GitHub file size limits training data is not available. If needed, please contact us.
