import pandas as pd
import numpy as np
from NewsScraping.Markets import Markets

openMarkets = Markets().getOpenMarkets()

print(openMarkets)