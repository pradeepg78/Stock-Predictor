import yfinance as yf

# ^GSPC is the ticker for the S&P 500
sp500 = yf.Ticker("^GSPC") 

sp500 = sp500.history(period="max")

