import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import random

warnings.filterwarnings('ignore')


class DataHandler:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {}
        self.factor_data = {}
        # 股票代码映射表，将SS/SZ后缀转换为正确的Yahoo Finance格式
        self.stock_code_mapping = {
            '600519.SS': '600519.SS',
            '000858.SZ': '000858.SZ',
            '601318.SS': '601318.SS',
            '600036.SS': '600036.SS',
            '000333.SZ': '000333.SZ',
            '600900.SS': '600900.SS',
            '601888.SS': '601888.SS',
            '600276.SS': '600276.SS',
            '600887.SS': '600887.SS',
            '601166.SS': '601166.SS',
            '601668.SS': '601668.SS',
            '601328.SS': '601328.SS',
            '601398.SS': '601398.SS',
            '601288.SS': '601288.SS',
            '601988.SS': '601988.SS'
        }

    def get_yahoo_code(self, stock_code):
        """获取Yahoo Finance格式的股票代码"""
        return self.stock_code_mapping.get(stock_code, stock_code)

    def get_stock_price(self, stock_code, start_date, end_date):
        """获取股票价格数据"""
        try:
            # 检查缓存
            key = f"{stock_code}_{start_date}_{end_date}"
            if key in self.stock_data:
                return self.stock_data[key]

            # 使用yfinance获取数据
            yahoo_code = self.get_yahoo_code(stock_code)
            df = yf.download(yahoo_code, start=start_date, end=end_date, progress=False)

            if df.empty:
                # 如果获取失败，生成模拟数据
                df = self.generate_mock_data(stock_code, start_date, end_date)
                print(f"使用模拟数据: {stock_code}")

            # 重命名列
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # 添加技术指标
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['rsi'] = self.calculate_rsi(df['close'])

            # 缓存数据
            self.stock_data[key] = df
            return df
        except Exception as e:
            print(f"获取股票价格失败: {stock_code}, {e}")
            # 生成模拟数据作为后备
            return self.generate_mock_data(stock_code, start_date, end_date)

    def generate_mock_data(self, stock_code, start_date, end_date):
        """生成模拟股票数据"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # 过滤掉周末
        dates = dates[dates.dayofweek < 5]

        # 生成随机价格数据
        np.random.seed(hash(stock_code) % 1000)  # 使用股票代码作为随机种子
        base_price = np.random.uniform(10, 100)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * (1 + returns).cumprod()

        # 生成OHLCV数据
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = prices * (1 + np.random.normal(0, 0.01, len(dates)))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        df['volume'] = np.random.lognormal(15, 1, len(dates))

        # 确保价格合理
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].abs()

        # 添加技术指标
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])

        return df

    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_fundamental_data(self, stock_code, date):
        """获取财务基本面数据（模拟数据）"""
        try:
            # 为每只股票生成固定的基本面数据
            if stock_code not in self.factor_data:
                # 创建一些随机但合理的基本面数据
                np.random.seed(hash(stock_code) % 1000)  # 使用股票代码作为随机种子
                self.factor_data[stock_code] = {
                    'market_cap': np.random.uniform(1e9, 1e11),
                    'roe': np.random.uniform(0.05, 0.25),
                    'pe_ratio': np.random.uniform(10, 30),
                    'pb_ratio': np.random.uniform(1, 5)
                }
            return self.factor_data[stock_code]
        except Exception as e:
            print(f"获取基本面数据失败: {stock_code}, {e}")
            return {'market_cap': 1e10, 'roe': 0.15, 'pe_ratio': 15, 'pb_ratio': 2}

    def get_trading_dates(self, start_date, end_date):
        """获取交易日期列表"""
        # 生成交易日历
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # 生成所有工作日
        all_days = pd.date_range(start, end, freq='D')

        # 过滤掉周末
        trading_dates = all_days[all_days.dayofweek < 5]

        return trading_dates

    def predict_price(self, stock_code, date):
        """预测股票价格（模拟预测）"""
        try:
            # 获取历史价格数据
            start_date = (pd.to_datetime(date) - timedelta(days=60)).strftime('%Y-%m-%d')
            price_data = self.get_stock_price(stock_code, start_date, date)
            if price_data.empty:
                return None

            # 使用简单的技术指标进行预测
            if len(price_data) < 20:
                return None

            current_price = price_data.iloc[-1]['close']
            ma5 = price_data.iloc[-1]['ma5']
            ma20 = price_data.iloc[-1]['ma20']
            rsi = price_data.iloc[-1]['rsi'] if not pd.isna(price_data.iloc[-1]['rsi']) else 50

            # 简单的预测逻辑
            if not pd.isna(ma5) and not pd.isna(ma20):
                if ma5 > ma20 and rsi < 70:  # 上涨趋势且未超买
                    predicted_change = np.random.uniform(0.01, 0.05)  # 预测上涨1-5%
                elif ma5 < ma20 and rsi > 30:  # 下跌趋势且未超卖
                    predicted_change = np.random.uniform(-0.05, -0.01)  # 预测下跌1-5%
                else:
                    predicted_change = np.random.uniform(-0.02, 0.02)  # 横盘震荡
            else:
                predicted_change = np.random.uniform(-0.03, 0.03)  # 随机波动

            predicted_price = current_price * (1 + predicted_change)
            return predicted_price
        except Exception as e:
            print(f"预测价格失败: {stock_code}, {e}")
            # 生成一个基于随机波动的预测
            price_data = self.get_stock_price(stock_code, date, date)
            if not price_data.empty:
                current_price = price_data.iloc[-1]['close']
                return current_price * (1 + np.random.uniform(-0.05, 0.05))
            return None