def getSustainability(stock):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np
    import math

    # Settiamo gli Headers

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    # Mettiamo su l'URL di base

    stock = stock

    target_url = "https://finance.yahoo.com/quote/" + stock + "/sustainability?p=" + stock

    # settiamo la ricerca con BeautifulSoup

    resp = requests.get(target_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    a_tag = soup.findAll('div')

    fs = list()
    for i in a_tag:
        fs.append(i)

    fs = pd.Series(fs).astype(str)

    base = (fs[(fs.str.contains('Environment Risk Score'))]).reset_index()
    del [base['index']]

    if base.empty == False:
        envT = base[0][1].find('Environment Risk Score')
        socT = base[0][1].find('Social Risk Score')
        govT = base[0][1].find('Governance Risk Score')
        overallT = base[0][1].find('Total ESG Risk score')

        envT1 = base[0][1].find('Environment Risk Score') + 138
        socT1 = base[0][1].find('Social Risk Score') + 133
        govT1 = base[0][1].find('Governance Risk Score') + 137
        overallT1 = base[0][1].find('Total ESG Risk score') + 98

        envScore = (base[0][1][envT1: envT1 + base[0][1][envT1:].find('</div')])
        socScore = (base[0][1][socT1: socT1 + base[0][1][socT1:].find('</div')])
        govScore = (base[0][1][govT1: govT1 + base[0][1][govT1:].find('</div')])
        overall = (base[0][1][overallT1: overallT1 + base[0][1][overallT1:].find('</div')])

        # crea il database

        oneStockData = pd.concat([pd.Series(stock), pd.Series(envScore), pd.Series(socScore), pd.Series(govScore),
                                  pd.Series(overall)], axis=1).set_axis(
            ['Stock', 'Environmental', 'Social', 'Governance',
             'Overall'], axis=1)

        return oneStockData

    if base.empty == True:
        return pd.DataFrame([])


def getGrowthEstimates(stock):
    # Routine di scraping per avere le raccomandation degli analisti con lo score relativo

    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np
    import math

    # Settiamo gli Headers

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    # Mettiamo su l'URL di base

    stock = stock

    target_url = "https://finance.yahoo.com/quote/" + stock + "/analysis?p=" + stock

    # settiamo la ricerca con BeautifulSoup

    resp = requests.get(target_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    a_tag = soup.findAll('div')

    fs = list()
    for i in a_tag:
        fs.append(i)

    fs = pd.Series(fs).astype(str)

    base = (fs[(fs.str.contains('Growth Estimate'))]).reset_index()
    del [base['index']]

    if base.empty == False:
        # In questa parte isoliamo le informazioni
        # Partiamo da IndexBase e tagliamo da qua

        indexBase = base[0][1][base[0][1].find('Growth Estimate'):]

        ixCQt = indexBase.find('class="Ta(end) Py(10px)">') + len('class="Ta(end) Py(10px)">')
        ixCNt = indexBase.find('Next Qtr.</span></td><td class="Ta(end) Py(10px)">') + len(
            'Next Qtr.</span></td><td class="Ta(end) Py(10px)">')
        ixCY = indexBase.find('Current Year</span></td><td class="Ta(end) Py(10px)">') + len(
            'Current Year</span></td><td class="Ta(end) Py(10px)">')
        ixNY = indexBase.find('Next Year</span></td><td class="Ta(end) Py(10px)">') + len(
            'Next Year</span></td><td class="Ta(end) Py(10px)">')
        ixN5Y = indexBase.find('Next 5 Years (per annum)</span></td><td class="Ta(end) Py(10px)">') + len(
            'Next 5 Years (per annum)</span></td><td class="Ta(end) Py(10px)">')

        # Definiamo i numeri tagliati

        growthCQt = indexBase[ixCQt: ixCQt + indexBase[ixCQt:].find('</td><td class')]
        growthNQt = indexBase[ixCNt: ixCNt + indexBase[ixCNt:].find('</td><td class')]
        growthCY = indexBase[ixCY: ixCY + indexBase[ixCY:].find('</td><td class')]
        growthNY = indexBase[ixNY: ixNY + indexBase[ixNY:].find('</td><td class')]
        growthN5Y = indexBase[ixN5Y: ixN5Y + indexBase[ixN5Y:].find('</td><td class')]

        # Mettiamo tutti insieme

        growthEstimates = pd.concat([pd.Series(growthCQt), pd.Series(growthNQt), pd.Series(growthCY),
                                     pd.Series(growthNY), pd.Series(growthN5Y)], axis=1).set_axis(['Current Quarter',
                                                                                                   'Next Quarter',
                                                                                                   'Current Year',
                                                                                                   'Next Year',
                                                                                                   'Next 5 Years (per annum)'],
                                                                                                  axis=1)

        growthEstimates = growthEstimates.set_index(pd.Series(stock))

        return growthEstimates

    if base.empty == True:
        return pd.DataFrame([])


def getOptions(stock):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np
    import math

    # Settiamo gli Headers

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    # Mettiamo su l'URL di base

    stock = stock

    target_url = "https://finance.yahoo.com/quote/" + stock + "/options?p=" + stock

    # settiamo la ricerca con BeautifulSoup

    resp = requests.get(target_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    a_tag = soup.findAll('div')

    fs = list()
    for i in a_tag:
        fs.append(i)

    fs = pd.Series(fs).astype(str)

    base = (fs[(fs.str.contains('Calls'))]).reset_index()
    del [base['index']]

    # Proviamo ad isolare tutti i core component delle stock options. In questo caso NON abbiamo una serie storica esatta.
    # Infatti, ogni opzione ha tutti i dati relativi dentro di se "insieme". Le strade sono due: prendere ogni componente
    # singolarmente, e concatenare, oppure prendere ogni singola Opzione, e concatenare verticalmente. Questa ultima strada
    # sembra molto più praticabile, perchè con il primo metodo rischieremo di fare un matching NON CORRETTO tra i diversi
    # componenti.

    # Scegliedo questa strada, prima sono da "Isolare" le diverse stringhe che riguardano ogni diversa opzione, e poi sono da
    # "Scorporare" tutti i singoli componenti. La vera challenge è quella di isolare le singole opzioni, perchè hanno un nome
    # che è unico e dipende dalla data. Vogliamo un codice che sia GENERALIZZABILE, quindi non si può fare un cerca su un
    # opzione semplice. Unica cosa che possiamo dare per certo: il nome di un'opzione è composto da: ticker + 15 cifre (che
    # indicano la data e altre cose).

    if base.empty == False:

        indexBase = base[0][1][base[0][1].find('Calls'):base[0][1].find('Puts')]

        # Andiamo a creare n (n = numero di opzioni) indexBase, uno per ogni opzione. per LIN sono 33 le Call

        indexBaseU = indexBase

        optString = list()

        # Inizializza le liste dei parametri

        name = list()
        strike = list()
        price = list()
        bid = list()
        ask = list()
        change = list()
        perChange = list()
        volume = list()
        openInterest = list()
        volatility = list()
        lastTrade = list()
        exDate = list()

        while len(optString) < 30:

            indexUpd = indexBaseU.find('href="/quote/' + stock + '2') + len('href="/quote/' + stock + '2')
            stringUpd = indexBaseU[indexUpd: indexUpd + indexBaseU[indexUpd:].find('Ell C($linkColor)')]

            indexBaseU = indexBaseU[indexUpd:]

            optString.append(stringUpd)

            # print(stringUpd)
            # print('\n')

            # Estraiamo i parametri

            # Estraiamo il nome dell'opzione: basta prendere da 'p=' a ' " '
            optionName = stringUpd[stringUpd.find('p=') + len('p='): stringUpd.find('"')]
            name.append(optionName)

            # Estraiamo lo strike: basta prendere da "strike=" a "&amp"
            strikePrice = stringUpd[stringUpd.find('strike=') + len('strike='): stringUpd.find('&amp')]
            strike.append(strikePrice)

            # Estraiamo il last price: basta prendere da "class="data-col3 Ta(end) Pstart(7px)">" a "</td><td class="data-col4"
            currentPrice = stringUpd[stringUpd.find('class="data-col3 Ta(end) Pstart(7px)">') +
                                     len('class="data-col3 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col4')]
            price.append(currentPrice)

            # Estraiamo Bid e Ask, con una procedura simile a quella di prima
            bidPrice = stringUpd[stringUpd.find('class="data-col4 Ta(end) Pstart(7px)">') +
                                 len('class="data-col4 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col5')]
            bid.append(bidPrice)
            askPrice = stringUpd[stringUpd.find('class="data-col5 Ta(end) Pstart(7px)">') +
                                 len('class="data-col5 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col6')]
            ask.append(askPrice)

            # Estraiamo il change e il %change

            changeP = stringUpd[stringUpd.find('class="data-col6 Ta(end) Pstart(7px)"><span class="">') +
                                len('class="data-col6 Ta(end) Pstart(7px)"><span class="">'): stringUpd.find(
                '</span></td><td class="data-col7')]
            if len(changeP) > 10:
                changeP = stringUpd[
                          stringUpd.find('"data-col6 Ta(end) Pstart(7px)"><span class="C($positiveColor)">+<!-- -->') +
                          len('"data-col6 Ta(end) Pstart(7px)"><span class="C($positiveColor)">+<!-- -->'): stringUpd.find(
                              '</span></td><td class="data-col7')]
                if len(changeP) > 10:
                    changeP = stringUpd[
                              stringUpd.find('class="data-col6 Ta(end) Pstart(7px)"><span class="C($negativeColor)">') +
                              len('class="data-col6 Ta(end) Pstart(7px)"><span class="C($negativeColor)">'): stringUpd.find(
                                  '</span></td><td class="data-col7')]

            change.append(changeP)

            perChangeP = stringUpd[stringUpd.find('class="data-col7 Ta(end) Pstart(7px)"><span class="">') +
                                   len('class="data-col7 Ta(end) Pstart(7px)"><span class="">'): stringUpd.find(
                '</span></td><td class="data-col8')]

            if len(perChangeP) > 10:
                perChangeP = stringUpd[
                             stringUpd.find(
                                 '"data-col7 Ta(end) Pstart(7px)"><span class="C($positiveColor)">+<!-- -->') +
                             len('"data-col7 Ta(end) Pstart(7px)"><span class="C($positiveColor)">+<!-- -->'): stringUpd.find(
                                 '</span></td><td class="data-col8')]
                if len(perChangeP) > 10:
                    perChangeP = stringUpd[stringUpd.find(
                        'class="data-col7 Ta(end) Pstart(7px)"><span class="C($negativeColor)">') +
                                           len('class="data-col7 Ta(end) Pstart(7px)"><span class="C($negativeColor)">'): stringUpd.find(
                        '</span></td><td class="data-col8')]

            perChange.append(perChangeP)

            # Volume
            volumeP = stringUpd[stringUpd.find('class="data-col8 Ta(end) Pstart(7px)">') +
                                len('class="data-col8 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col9')]
            volume.append(volumeP)

            # Open Interest
            openInterestP = stringUpd[stringUpd.find('class="data-col9 Ta(end) Pstart(7px)">') +
                                      len('class="data-col9 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col10')]
            openInterest.append(openInterestP)

            # Implied Volatility
            impliedVol = stringUpd[stringUpd.find('class="data-col10 Ta(end) Pstart(7px) Pend(6px) Bdstartc(t)">') +
                                   len('class="data-col10 Ta(end) Pstart(7px) Pend(6px) Bdstartc(t)">'): stringUpd.find(
                '</td></tr><tr class="data-row')]
            volatility.append(impliedVol)

            # Last Trade Date
            lastTradeDate = stringUpd[stringUpd.find('class="data-col1 Ta(end) Pstart(7px)">') +
                                      len('class="data-col1 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col2')][0:10]
            lastTrade.append(lastTradeDate)

            # Prendiamo la exercise date direttamente dal nome dell'opzione
            optionNameForDate = optionName[optionName.find('2'):]
            exYear = optionNameForDate[0:2]
            exMonth = optionNameForDate[2:4]
            exDay = optionNameForDate[4:6]
            exDateP = '20' + exYear + '-' + exMonth + '-' + exDay

            exDate.append(exDateP)

        # uniamo tutto

        callOptions = pd.concat([pd.Series(exDate), pd.Series(name), pd.Series(strike), pd.Series(price),
                                 pd.Series(bid), pd.Series(ask), pd.Series(change), pd.Series(perChange),
                                 pd.Series(volatility), pd.Series(lastTrade), pd.Series(openInterest),
                                 pd.Series(volume)],
                                axis=1).set_axis(['Expiration Date',
                                                  'Option Code', 'Strike Price', 'Current Option Price', 'Bid', 'Ask',
                                                  'Change', '% Change',
                                                  'Implied Volatility (%)', 'Last Trade Date', 'Open Interest',
                                                  'Volume'],
                                                 axis=1)

        callOptions = pd.concat(
            [pd.DataFrame(np.full(len(callOptions['Open Interest']), 'CALL')).set_axis(['Side'], axis=1), callOptions],
            axis=1)

        callOptions = callOptions[callOptions['Option Code'].str.contains(stock)].reset_index()
        del [callOptions['index']]

        # Proviamo a fare lo stesso per le put Options

        putSample = base[0][1][base[0][1].find('Puts'):]

        optString = list()

        # Inizializza le liste dei parametri

        name = list()
        strike = list()
        price = list()
        bid = list()
        ask = list()
        change = list()
        perChange = list()
        volume = list()
        openInterest = list()
        volatility = list()
        lastTrade = list()
        exDate = list()

        while len(optString) < 30:

            indexUpd = indexBaseU.find('href="/quote/' + stock + '2') + len('href="/quote/' + stock + '2')
            stringUpd = indexBaseU[indexUpd: indexUpd + indexBaseU[indexUpd:].find('Ell C($linkColor)')]

            indexBaseU = indexBaseU[indexUpd:]

            optString.append(stringUpd)

            # print(stringUpd)
            # print('\n')

            # Estraiamo i parametri

            # Estraiamo il nome dell'opzione: basta prendere da 'p=' a ' " '
            optionName = stringUpd[stringUpd.find('p=') + len('p='): stringUpd.find('"')]
            name.append(optionName)

            # Estraiamo lo strike: basta prendere da "strike=" a "&amp"
            strikePrice = stringUpd[stringUpd.find('strike=') + len('strike='): stringUpd.find('&amp')]
            strike.append(strikePrice)

            # Estraiamo il last price: basta prendere da "class="data-col3 Ta(end) Pstart(7px)">" a "</td><td class="data-col4"
            currentPrice = stringUpd[stringUpd.find('class="data-col3 Ta(end) Pstart(7px)">') +
                                     len('class="data-col3 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col4')]
            price.append(currentPrice)

            # Estraiamo Bid e Ask, con una procedura simile a quella di prima
            bidPrice = stringUpd[stringUpd.find('class="data-col4 Ta(end) Pstart(7px)">') +
                                 len('class="data-col4 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col5')]
            bid.append(bidPrice)
            askPrice = stringUpd[stringUpd.find('class="data-col5 Ta(end) Pstart(7px)">') +
                                 len('class="data-col5 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col6')]
            ask.append(askPrice)

            # Estraiamo il change e il %change

            changeP = stringUpd[stringUpd.find('class="data-col6 Ta(end) Pstart(7px)"><span class="">') +
                                len('class="data-col6 Ta(end) Pstart(7px)"><span class="">'): stringUpd.find(
                '</span></td><td class="data-col7')]
            if len(changeP) > 10:
                changeP = stringUpd[
                          stringUpd.find('"data-col6 Ta(end) Pstart(7px)"><span class="C($positiveColor)">+<!-- -->') +
                          len('"data-col6 Ta(end) Pstart(7px)"><span class="C($positiveColor)">+<!-- -->'): stringUpd.find(
                              '</span></td><td class="data-col7')]

            change.append(changeP)

            perChangeP = stringUpd[stringUpd.find('class="data-col7 Ta(end) Pstart(7px)"><span class="">') +
                                   len('class="data-col7 Ta(end) Pstart(7px)"><span class="">'): stringUpd.find(
                '</span></td><td class="data-col8')]

            if len(perChangeP) > 10:
                perChangeP = stringUpd[
                             stringUpd.find(
                                 '"data-col7 Ta(end) Pstart(7px)"><span class="C($positiveColor)">+<!-- -->') +
                             len('"data-col7 Ta(end) Pstart(7px)"><span class="C($positiveColor)">+<!-- -->'): stringUpd.find(
                                 '</span></td><td class="data-col8')]

            perChange.append(perChangeP)

            # Volume
            volumeP = stringUpd[stringUpd.find('class="data-col8 Ta(end) Pstart(7px)">') +
                                len('class="data-col8 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col9')]
            volume.append(volumeP)

            # Open Interest
            openInterestP = stringUpd[stringUpd.find('class="data-col9 Ta(end) Pstart(7px)">') +
                                      len('class="data-col9 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col10')]
            openInterest.append(openInterestP)

            # Implied Volatility
            impliedVol = stringUpd[stringUpd.find('class="data-col10 Ta(end) Pstart(7px) Pend(6px) Bdstartc(t)">') +
                                   len('class="data-col10 Ta(end) Pstart(7px) Pend(6px) Bdstartc(t)">'): stringUpd.find(
                '</td></tr><tr class="data-row')]
            volatility.append(impliedVol)

            # Last Trade Date
            lastTradeDate = stringUpd[stringUpd.find('class="data-col1 Ta(end) Pstart(7px)">') +
                                      len('class="data-col1 Ta(end) Pstart(7px)">'): stringUpd.find(
                '</td><td class="data-col2')][0:10]
            lastTrade.append(lastTradeDate)

            # Prendiamo la exercise date direttamente dal nome dell'opzione
            optionNameForDate = optionName[optionName.find('2'):]
            exYear = optionNameForDate[0:2]
            exMonth = optionNameForDate[2:4]
            exDay = optionNameForDate[4:6]
            exDateP = '20' + exYear + '-' + exMonth + '-' + exDay

            exDate.append(exDateP)

        putOptions = pd.concat([pd.Series(exDate), pd.Series(name), pd.Series(strike), pd.Series(price),
                                pd.Series(bid), pd.Series(ask), pd.Series(change), pd.Series(perChange),
                                pd.Series(volatility), pd.Series(lastTrade), pd.Series(openInterest),
                                pd.Series(volume)],
                               axis=1).set_axis(['Expiration Date',
                                                 'Option Code', 'Strike Price', 'Current Option Price', 'Bid', 'Ask',
                                                 'Change', '% Change',
                                                 'Implied Volatility (%)', 'Last Trade Date', 'Open Interest',
                                                 'Volume'],
                                                axis=1)
        putOptions = pd.concat(
            [pd.DataFrame(np.full(len(putOptions['Open Interest']), 'PUT')).set_axis(['Side'], axis=1),
             putOptions], axis=1)

        putOptions = putOptions[putOptions['Option Code'].str.contains(stock)].reset_index()
        del [putOptions['index']]

        # Uniamo

        totalOption = pd.concat([callOptions, putOptions], axis=0).reset_index()
        del [totalOption['index']]

        totalOption = pd.concat([pd.DataFrame(np.full(len(totalOption['Side']), stock)).set_axis(['Ticker'], axis=1),
                                 totalOption], axis=1)

        return totalOption

    if base.empty == True:
        # print ('No Data Available on Yahoo Finance')

        return pd.DataFrame([])
