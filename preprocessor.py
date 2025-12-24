from datetime import datetime
import numpy as np
import pandas as pd
import re

class PreProcessor():
    def __init__(self):
        self.bankruptcy_config = {
            'PERMNO': 'Int64',
            'B_date': 'string'
        }
        self.crsp_config = {
            'DATE': 'string',      
            'COMNAM': 'string', 
            'PERMNO': 'Int64', 
            'PERMCO': 'Int64', 
            'SHRCD': 'Int64',  
            'SICCD': 'string',  
            'CUSIP': 'string',
            'BIDLO': 'float64',
            'ASKHI': 'float64',
            'PRC': 'float64',
            'VOL': 'float64',
            'BID': 'float64',
            'ASK': 'float64',
            'SHROUT': 'Int64', 
            'RET': 'string',   
            'RETX': 'string',      
            'vwretd': 'float64'
        }
        
    def clean_numeric(self, x):
        """Remove any characters in column and convert to float"""
        if pd.isna(x): # type: ignore
            return np.nan
        cleaned = re.sub(r'[^0-9eE\.\-+]', '', str(x))
        try:
            return float(cleaned)
        except:
            return np.nan
    
    def customImputation(self, df, col:str, impute_type:str):
        df[col + "_imputed_flag"] = df[col].isna().astype(int)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
        if impute_type == "zero":
            df[col] = df[col].fillna(0)

        elif impute_type == "firm_mean":
            if "PERMNO" not in df.columns:
                raise ValueError("firm_mean imputation requires a PERMNO column.")
            df[col] = df.groupby("PERMNO")[col].transform(lambda s: s.fillna(s.mean()))
            df[col] = df[col].fillna(0)
        return df
    
    def read_data(self):
        bankruptcy = pd.read_csv('data/BANKRUPTCY.csv', dtype = self.bankruptcy_config, parse_dates=['B_date']) # type: ignore
        compustat = pd.read_csv('data/COMPUSTAT.csv')
        crsp = pd.read_csv('data/MSF.csv', dtype = self.crsp_config, parse_dates=['B_date'])  # type: ignore
        
        bankruptcy['bk_year'] = bankruptcy['B_date'].dt.year # type: ignore
        
        compustat['datadate'] = pd.to_datetime(compustat['datadate'], errors='coerce')
        compustat['year'] = compustat['datadate'].dt.year # type: ignore
        selected_features = ["year", "cusip", "sale", "gp", "ebit", "ni", "xint","at", "lt", "ceq", "che", 
                            "invt", "rect", "dltt","act", "lct", "oancf", "capx", "fincf", "ivncf", "dv"]
        compustat = compustat[selected_features].dropna(subset='year')
        
        crsp['RET'] = crsp['RET'].apply(self.clean_numeric)
        crsp['DATE'] = pd.to_datetime(crsp['DATE'], format='%Y%m%d', errors='coerce')
        crsp['year'] = crsp['DATE'].dt.year # type: ignore
        crsp_yearly = (
            crsp.sort_values(['PERMNO','DATE'])
            .groupby(['PERMNO','year'])
            .agg(
                PRC=('PRC', 'last'),
                VOL=('VOL', 'sum'),
                RET=('RET', 'last'),
                vwretd=('vwretd', 'mean'),
                SHROUT=('SHROUT', 'last')
            )
            .reset_index()
        )
        
        def cusip6(s):
            """Returns first 6 characters of CUSIP"""
            return s.astype("string").str.strip().str.upper().str[:6]

        crsp_yearly['CUSIP6'] = cusip6(crsp['CUSIP'])
        compustat['CUSIP6'] = cusip6(compustat['cusip'])
        
        compustat_lag = compustat[['CUSIP6','year'] + [c for c in compustat.columns if c not in {'CUSIP6','year'}]]
        compustat_lag['year'] = compustat_lag['year'] + 1
        crsp_lag = crsp_yearly.copy()
        crsp_lag["year"] = crsp_lag["year"] + 1
        
        df = pd.merge(compustat_lag,crsp_lag,on=['CUSIP6','year'],how='outer',validate='m:m')
        df = pd.merge(df, bankruptcy,left_on=['PERMNO','year'], right_on=['PERMNO','bk_year'], how='left')
        df['bankruptcy'] = (df['bk_year'].notna()).astype(int)
        df = df.drop(columns=['bk_year', 'B_date', 'cusip'])
        df = df.dropna(subset=['year', 'PERMNO'])
        required = ['PRC', 'RET','vwretd','at','lt','ceq', 'dltt','sale','ni','gp','che','xint','rect','invt','ebit','lct','capx','act']
        df = df.dropna(subset=required)
        
        df = self.customImputation(df, 'oancf', 'firm_mean')
        df = self.customImputation(df, 'fincf', 'firm_mean')
        df = self.customImputation(df, 'ivncf', 'firm_mean')
        df = self.customImputation(df, 'dv', 'zero')
        return df