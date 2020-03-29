import tushare as ts
ts.set_token('49aa7fe6d24f22ba78b6f47c064ffdddc3bc2ca8870a37cf8ead987b')
import pandas as pd
def get_stock_data(from_date,to_date,stock_code):

    pro = ts.pro_api()

    df = pro.daily(ts_code='600998.SH', start_date='20180701', end_date='20180718')
    #df = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
    #df = pro.index_daily(ts_code='600000.SH', start_date='20180701', end_date='20180718')
    df = ts.get_k_data(code='sh',ktype='D',autype='qfq',start='1900-12-20')
    df.to_csv('600000.SH.csv',index=None)
    return df

if __name__ == '__main__':
    data = get_stock_data(1,1,1)
    print(data)
    #stock = ts.get_today_all()
    #print(stock)