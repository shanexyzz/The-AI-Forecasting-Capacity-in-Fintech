import wrds
import pandas as pd

# Connect to WRDS
db = wrds.Connection()

def check_compustat_lags(tickers, year=2024):
    """
    Checks the reporting lag between Period End (datadate) 
    and Announcement Date (rdq).
    """
    # Converting tickers to a SQL-friendly string
    ticker_str = "'" + "','".join(tickers) + "'"
    
    # Query focusing on the date fields:
    # datadate: Fiscal Period End Date
    # rdq: Report Date of Quarterly Earnings (The "Announcement" date)
    # apdedate: Actual Period End Date
    sql_query = f"""
        SELECT gvkey, tic, datadate, rdq, fyearq, fqtr
        FROM comp.fundq
        WHERE tic IN ({ticker_str})
        AND fyearq = {year}
        AND indfmt = 'INDL' 
        AND datafmt = 'STD'
        AND popsrc = 'D' 
        AND consol = 'C'
    """
    
    data = db.raw_sql(sql_query, date_cols=['datadate', 'rdq'])
    
    # Calculate the actual lag in days
    data['reporting_lag_days'] = (data['rdq'] - data['datadate']).dt.days
    
    return data

# Example: Check for some major tickers
tickers_to_check = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
df_lags = check_compustat_lags(tickers_to_check)

print(df_lags.sort_values(['tic', 'datadate']))

# Check how many rows are missing the 'rdq'
missing_rdq = df_lags['rdq'].isnull().sum()
print(f"\nMissing Announcement Dates: {missing_rdq} out of {len(df_lags)}")