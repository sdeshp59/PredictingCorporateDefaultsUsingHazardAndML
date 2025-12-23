# Predicting Corporate Defaults Using Hazard and ML

Before running code, make sure you have the following:
- CRSP Dataset downloaded into data folder in directory
- COMPUSTAT Dataset downloaded into data folder in directory
- Bankruptcy Dataset downloaded into data folder in directory

To run the code, follow the following steps in a bash terminal:
1. Create virtual environment and activate by running: 

```bash
python3 -m venv venv && source venv/bin/activate
```

2. Upgrade pip and install all required libraries into virtual environment:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

3. Run the code:

```bash
python main.py
```

Outputs should appear in outputs folder and/or in your terminal