Note if you get stuck anywhere and it takes way too long then Cntrl + C and then run this in terminal: py --version. It has to say: Python 3.12.10

How to get this project to work:
Verify Python Version: py --version
You need to use python 3.12.10 for all the features to work because 3.14 doesn't support the wheels/setup tools.
If you don’t have Python 3.12 installed: winget install -e --id Python.Python.3.12
Created a new venv. Run in the terminal: py -3.12 -m venv venv
Activate the venv: .\venv\Scripts\activate
The front of the path has to say (verv)
Upgraded pip/setuptools/wheel inside venv. Run in the terminal: python -m pip install --upgrade pip setuptools wheel (this may take a while)
Next run this: pip install pandas==2.2.2
Install core packages. Run in the terminal: pip install -r requirements.txt (will mostly also take a while)
Install extras for visualization. Run in terminal: pip install matplotlib seaborn
Inside your activated venv, run: pip list ( makes sure you have everything in requirements will have extras)
If AAPL_2018_2023.csv is not there then run: python save_dataset.py (this is a dummy dataset)
Then you can finally run ter app: python app.py

