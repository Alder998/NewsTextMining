from IPython.display import clear_output
import NewsDownloadLibrary as nl

Bing = nl.MassiveNewsScaper (numberOfRandomStocks = 50, source = 'Bing', update_returns = True)
marketWatch = nl.MassiveNewsScaper (numberOfRandomStocks = 50, source = 'MarketWatch', update_returns = True)
CNBC = nl.MassiveNewsScaper (numberOfRandomStocks = 50, source = 'CNBC', update_returns = True)

print(CNBC)