import pandas as pd
import numpy as np
import datetime

TRADE_STRAGE = pd.DataFrame({'low':[-1,   -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,   0.1,     0.2,  0.3, 0.4,    0.5,   0.6,   0.7,  0.8, 0.9,  1,   2],
                             'high':[-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.2,     0.3,  0.4, 0.5,    0.6,   0.7,   0.8,  0.9, 1,    2,   3],
                             'ratio':[1,     0.9,  0.8,  0.7,  0.6,  0.5,  0.4,  0.3,  0.2, -0.05,-0.075, -0.1, -0.2,-0.25,   -0.3, -0.4, -0.5, -0.6,-0.7, -0.9]})

def calc_trend_of_investment(start_date,init_amount,trade_amount,trade_days,benchmark_mean_days,
                             trade_frequency,sale_start_point):
    """
    假设在开始日期买入初始本金的上证指数，以benchmark_mean_days天的均线作为交易判断的基准，
    交易的比率按TRADE_STRAGE设置的比率来执行，交易的频率为1月trade_frequency次
    @:param start_date 起始日期
    @:param init_amount 账户本金
    @:param trade_amount 定投金额
    @:param benchmark_mean_days N天均值作为交易基准
    @:param trade_frenquency 交易频率
    @:param sale_start_point 当前价格和均值比值高于该点时卖出
    @:param trade_days 交易天数
    """

    stock_quantity = 0 #股票数量
    cash_balance = init_amount #现金
    profit = 0 #收益
    current_price = 0
    last_trade_date = start_date
    stock_data = pd.read_csv('600000.SH.csv')

    for i in range(trade_days):
        date_index = stock_data[stock_data['date'] == start_date]
        dt = datetime.datetime.strptime(start_date, "%Y-%m-%d") #当前日期
        if len(date_index) > 0:
            date_index = stock_data.index.get_loc(stock_data[stock_data['date'] == start_date].index[0])
            mean_price = np.mean(stock_data.iloc[date_index - benchmark_mean_days:date_index]['close'])
            current_price = stock_data.iloc[date_index]['open']
            ratio = (current_price - mean_price) / mean_price
            trade_radio = TRADE_STRAGE[TRADE_STRAGE['low'] <= ratio][TRADE_STRAGE['high'] > ratio]['ratio']
            #print(trade_radio)
            #print(trade_radio.size)
            if trade_radio.size == 1:
                trade_radio = trade_radio.iloc[0]
                #比值比率在设定范围内，进行交易
                buy_amount = trade_amount * trade_radio

                last_trade_date_dt = datetime.datetime.strptime(last_trade_date, "%Y-%m-%d")
                #计算本次交易日期，如果当前日期在当天或之后可以交易
                trade_dt = last_trade_date_dt + datetime.timedelta(days=int(30 / trade_frequency))

                can_sale = stock_quantity > -1 * buy_amount / current_price #是否能卖出
                if cash_balance > buy_amount and (dt - trade_dt).days >= 0 and can_sale and ((trade_radio <= sale_start_point) or trade_radio > 0):
                    #交易
                    last_trade_date = start_date
                    cash_balance -= buy_amount
                    stock_quantity += buy_amount / current_price
                    print('交易时间%s：价格均值%d,交易金额：%d，当前股票价格:%d,账户总股票金额：%f,账户现金%d'
                    % (
                    start_date, mean_price, buy_amount, current_price, stock_quantity * current_price, cash_balance))

        start_date = (dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")  # 更新起始日期,加1天

    print(stock_quantity * current_price + cash_balance)



if __name__ == '__main__':
    calc_trend_of_investment('2010-01-04',1000000,200000,356 * 11,500,3,-0.2)