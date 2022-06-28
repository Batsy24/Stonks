import kivy
import json
import yfinance as yf
import chartData as returndata
import stonksDataRetriever as sdr
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from matplotlib import pyplot as plt
from kivy.garden.matplotlib import FigureCanvasKivyAgg as fcka

factor = 2 # 1.6 for original image size
Window.size = (1458/factor, 1600/factor)

Builder.load_file('stonkstrial2screenmanager.kv')

infovals1 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
infovals2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
infovals3 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
infovals4 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
infovals5 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

time = '3mo'
ticker = 'gs'
column = 'Close'
window = 200
stock_news = []
car1num = 0
car2num = 0
car3num = 0
car4num = 0
car5num = 0

class HomePage(Screen):
    def __init__(self, **kwargs):
        super(HomePage, self).__init__(**kwargs)
        
    def addInCarousel1(self):
        global ticker
        global car1num
        global time
        global column
        global infovals1
        global window
        global stock_news
        window_period = self.ids.input_window.text
        if window_period == '' or window_period == 'Enter Period':
            self.ids.input_window.text = '200'
            window = 200
        else:
            window = int(window_period)
            
        ticker = self.ids.input_text.text
        if ticker == '' or ticker == 'Enter Input Ticker':
            self.ids.input_text.text = 'Error'
        elif car1num < 1 and (ticker != '' and ticker != 'Error'):
            fig, ax = plt.subplots()
            dataset_main = returndata.give_chart_data(time, ticker)
            if column == 'Close':
                dataset = dataset_main[column]
                ax.plot(dataset)
                ax.set_title(f'{ticker}, {time}, Close')
                
            elif column == 'MA':
                dataset1 = dataset_main['Close']
                dataset2 = returndata.MA(window, dataset_main)
                ax.plot(dataset1, alpha=0.5)
                ax.plot(dataset2)
                ax.set_title(f'{ticker}, {time}, Close, MA')
            elif column == 'EMA':
                dataset1 = dataset_main['Close']
                dataset2 = returndata.EMA(window, dataset_main)
                ax.plot(dataset1, alpha=0.5)
                ax.plot(dataset2)
                ax.set_title(f'{ticker}, {time}, Close, EMA')
            elif column == 'MACD':
                macd_data = returndata.MACD(dataset_main)
                shape_0 = macd_data.shape[0]
                xmacd_ = shape_0 - len(macd_data)

                macd_data = macd_data.iloc[-len(macd_data):, :]
                ax.plot(macd_data['Close'], label='Closing Price')                
                ax.set_title(f'{ticker}, {time}, Close, MACD')
                ax.grid()
                self.graph1 = fcka(plt.gcf())
                self.ids.graph_box1.add_widget(self.graph1)
                self.graph1.size_hint = (1, 0.8)

                
                fig1, ax1 = plt.subplots()
                ax1.plot(macd_data['MACD'])
                ax1.plot(macd_data['signal_line'])
                self.macd_graph = fcka(plt.gcf())
                self.ids.graph_box1.add_widget(self.macd_graph)
                self.macd_graph.size_hint = (1, 0.2)
                car1num += 2
            elif column == 'Bollinger':
                dataset = returndata.Bollinger(dataset_main)
                last_days = len(dataset)
                fig.dpi = 100
                shape_0 = dataset.shape[0]
                xmacd_ = shape_0 - last_days

                dataset = dataset.iloc[-last_days:, :]
                x_ = range(3, dataset.shape[0])
                x_ = list(dataset.index)

                ax.plot(dataset['BOLU'], label='Upper Band', color='c')
                ax.plot(dataset['BOLD'], label='Lower Band', color='c')
                ax.fill_between(x_, dataset['BOLD'], dataset['BOLU'], alpha=0.35)

                ax.plot(dataset['Close'], label='Closing Price', color='b', alpha=0.25)
                ax.set_title(f'{ticker}, {time}, Close, Bollinger Bands')
                ax.set_ylabel('USD')

                ax.legend()


            if column != 'MACD':
                ax.grid()
                    
                self.graph1 = fcka(plt.gcf())
                self.ids.graph_box1.add_widget(self.graph1)
                #self.ids.input_text.text = ''
                car1num += 1

            news_object = sdr.return_news_object(ticker)
            links, news_key = tuple(sdr.get_news(news_object))
            samples = tuple(sdr.get_news_sample(links))
            news_samples = sdr.get_trimmed_news(samples)
            
           
            infovals1, stock_news = sdr.get_data(news_key, news_samples, ticker)
            print(stock_news)
            
            # print(infovals1)

            infostr1 = f'''\r
Open: {infovals1['open']}          Mkt Cap: {infovals1['marketCap']}          Prev. Close: {infovals1['previousClose']}
High: {infovals1['dayHigh']}          P.E: {infovals1['trailingPE']}                  52 Wk. High: {infovals1['fiftyTwoWeekHigh']}
Low: {infovals1['dayLow']}              Div Yield: {infovals1['dividendYield']}        52 Wk Low: {infovals1['fiftyTwoWeekLow']}'''
            self.ids.graph_label1.text = infostr1 

            self.ids.heading_label.text = f"{ticker.upper()}"
            self.ids.longName.text = f'{infovals1["longName"]}       {infovals1["financialCurrency"]}'
            difference = infovals1['open'] - infovals1['previousClose']
            if difference >= 0:
                self.ids.difference.color = (34/255, 139/255, 34/255, 1)
            else:
                self.ids.difference.color = (1, 0, 0, 1)
            self.ids.difference.text = str(abs(difference))
    

    

    def removeFromCarousel1(self):
        global car1num
        global column
        if car1num ==1:
            self.ids.graph_box1.remove_widget(self.graph1)
            self.ids.heading_label.text = ''
            self.ids.longName.text = ''
            self.ids.difference.text = ''
            car1num -= 1
        elif car1num == 2 and column == 'MACD':
            self.ids.graph_box1.remove_widget(self.graph1)
            self.ids.graph_box1.remove_widget(self.macd_graph)
            self.ids.heading_label.text = ''
            self.ids.longName.text = ''
            self.ids.difference.text = ''
            
            car1num -= 2
        
    


        
    def set_time1(self, instance, boolean, curtime):
        global time
        if boolean:
            time = curtime

    def set_col1(self, instance, boolean, curcolumn):
        global column
        if boolean:
            column = curcolumn
            
        
        
    
class ForecastPage(Screen):
    def __init__(self, **kwargs):
        super(ForecastPage, self).__init__(**kwargs)

class NewsPage(Screen):
    def __init__(self, **kwargs):
        super(NewsPage, self).__init__(**kwargs)

    def News(self):
        for i in range(len(stock_news) - 1):
            self.ids.news_label.text += f'{i + 1}.'
            for item in stock_news[i]:
                
                self.ids.news_label.text += item + '\n\n'
        
    
class MeraApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomePage(name='HomePage'))
        sm.add_widget(ForecastPage(name='ForecastPage'))
        sm.add_widget(NewsPage(name='NewsPage'))
        return sm


MeraApp().run()


