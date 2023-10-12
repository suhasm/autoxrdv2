# AutoXRD


## GSAS-II Installation 
For Intel Mac, follow the guide from GSAS-II website: 
- https://subversion.xray.aps.anl.gov/trac/pyGSAS/wiki/InstallMac
- https://subversion.xray.aps.anl.gov/trac/pyGSAS/wiki/MacSingleStepInstallerFigs (with screen shot)
- 
## Installation on Intel Mac
- This process worked for installing on Intel Mac Ventura 13.4.1
- Run ```g2="https://subversion.xray.aps.anl.gov/admin_pyGSAS/downloads/gsas2full-Latest-MacOSX-x86_64.sh"
curl "$g2" > /tmp/g2.sh; bash /tmp/g2.sh -b -p ~/g2full``` in your terminal and install in default location. Leave the proxy blank.
- Create a new environment with python 3.10, and install numpy, scipy, wxpython, matplotlib, pandas:
  ```
  conda create --name myenv python=3.10
  conda activate myenv
  conda install numpy scipy matplotlib pandas
  pip install wxpython
  ```
- Run ```import sys, os``` ```sys.path.insert(0, "/Users/your_username/g2full/GSASII")``` ```import GSASIIscriptable as G2sc```
- It should import successfully

## Testing RangeDetermine.ipynb

- Make you set the system path to include the GSAS installation. Otherwise you won't be able to access it:
  ```
  import sys
  sys.path.append('/Users/suhasmahesh/g2full/GSASII')
  ```
- You will also have to install a bunch of libraries for it to work:
  - ```pip install numpyro jax jaxlib```
- You may also have to delete ```EFtables = calcControls['EFtables']``` from GSASIIstrMath.py
- You may also have to add the GSAS installation to the system path in gsas.py:
  ```
  gsas_bin = os.path.join(sys.base_prefix, "/Users/suhasmahesh/g2full/GSASII/bindist")
  gsas_path = os.path.join(sys.base_prefix, "/Users/suhasmahesh/g2full/GSASII")
  ``
## Note
- .gpx is the GSAS-II project file, we import this GSAS project to our script
- .yaml is the anaconda environment file; it has the python packages needed to run the script
- b2 folder contains the necessary functions in .py scripts
- TrialRangeData folder contains the csv files of simulated data to be run in the 1_Inference.ipynb notebook
- Bkg substraction / Peak fitting: https://github.com/fang-ren/Discover_MG_CoVZr/tree/master/scripts/XRD_processing
