# this script gives you news and all the home page data except for the data for the graph

import yfinance as yf
import requests
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import json


# THINGS I STILL NEED TO GET
# + - values
# % values

ticker_input = 'gs'

# noinspection PyBroadException
try:
    ticker = ticker_input
    stock = yf.Ticker(ticker)
    news_ = stock.news
except:
    print('invalid ticker')


def get_news(news_list=news_):
    link_list = list()
    news_keys = list()
    for i in news_list:
        link = i['link']
        link_list.append(link)
        title = i['title']
        news_keys.append(title)

    return link_list, news_keys


# noinspection PyBroadException
def get_news_sample(link_list):
    samples_list = list()
    for link in link_list:
        data = requests.get(link).text
        div_caas_body = SoupStrainer('div', {"class": "caas-body"})
        scraper = BeautifulSoup(data, 'lxml', parse_only=div_caas_body)
        try:
            for i in scraper:
                txt = scraper.find('p').text
            samples_list.append(txt)
        except:
            placeholder = ''
            samples_list.append(placeholder)

    return samples_list


def get_trimmed_news(sample_list):
    new_samples = list()
    str_slice = slice(0, 120)
    for sample in sample_list:
        trimmed_sample = sample[str_slice]
        new_sample = trimmed_sample + '... click to read more'
        new_samples.append(new_sample)

    return new_samples


# noinspection PyBroadException
def get_data(news_titles, news_list):
    stats = dict()
    news = list()
    stat_keys = ['sector', 'longBusinessSummary', 'website', 'industry', 'currentPrice', 'financialCurrency',
                 'longName', 'symbol', 'previousClose', 'open', 'trailingPE', 'marketCap', 'dayLow', 'fiftyTwoWeekHigh',
                 'fiftyTwoWeekLow', 'dividendYield', 'dayHigh', 'logo_url']

    info = stock.info
    stats.fromkeys(stat_keys, None)

    for key in stat_keys:
        stats[key] = info[key]

    for i in range(len(news_titles)):
        news_pair = (news_titles[i], news_list[i])
        news.append(news_pair)

    return stats, news


links, news_key = tuple(get_news())
samples = tuple(get_news_sample(links))
news_samples = get_trimmed_news(samples)

# noinspection PyBroadException
try:
    stock_stats, stock_news = get_data(news_key, news_samples)
    json_dat = [stock_stats, stock_news]
    with open('data.json', 'w', encoding='utf-8') as file:
        json.dump(json_dat, file, ensure_ascii=False, indent=4)
except:
    print('invalid ticker, please try again')

# ------------------------------------------------------------------------------------------
# Example code to read from the json

# with open('data.json', 'r', encoding='utf-8') as file:
    # data = json.load(file)
# -------------------------------------------------------------------------------------------

# the data you receive will be in the format of a list of 2 elements

# i) the first element will be a dictionary which holds basic stock info to display

# ii) the second element will be a list in which there's tuples of which the first element will be a news headline
# and the second element will be a news sample text corresponding to the headline
