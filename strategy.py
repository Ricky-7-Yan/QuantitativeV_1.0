import numpy as np
import pandas as pd
import config
import data_utils


class MultiFactorStrategy:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.config = config.STRATEGY_CONFIG
        self.current_date = None
        self.portfolio = {
            'cash': self.config['capital_base'],
            'positions': {},  # {stock: {'shares': num, 'avg_price': price}}
            'value': self.config['capital_base']
        }
        self.trade_history = []
        self.if_trade = False
        self.t = 0
        self.all_stocks = []  # 股票池
        self.current_portfolio = []  # 当前持仓组合
        self.logs = []  # 日志记录

    def log_message(self, message):
        """记录日志"""
        self.logs.append(f"{self.current_date}: {message}")

    def initialize(self):
        """初始化策略"""
        self.set_params()
        self.set_variables()
        self.log_message("策略初始化完成")

    def set_params(self):
        """设置策略参数"""
        self.tc = self.config['rebalance_freq']
        self.yb = self.config['sample_period']
        self.N = self.config['hold_num']
        self.factors = self.config['factors']
        self.weights = self.config['factor_weights']

    def set_variables(self):
        """设置中间变量"""
        self.t = 0
        self.if_trade = False

    def before_trading_start(self, date):
        """每日开盘前处理"""
        self.current_date = date
        if self.t % self.tc == 0:
            self.if_trade = True
            # 获取可行股票池
            self.all_stocks = config.STOCK_POOL
            self.log_message(f"调仓日，股票池: {len(self.all_stocks)}只股票")
        self.t += 1

    def handle_data(self):
        """每日交易处理"""
        if not self.if_trade:
            return

        # 计算每只股票分配的资金
        self.every_stock = self.portfolio['value'] / self.N

        # 获取因子数据并排序
        factor_scores, stock_codes = self.get_ranked_factors()

        # 计算综合得分
        points = np.dot(factor_scores, self.weights)

        # 对股票按得分排序
        sorted_indices = np.argsort(points)[::-1]  # 降序排序
        sorted_stocks = [stock_codes[i] for i in sorted_indices]

        # 取前N名作为买入股票
        to_buy = sorted_stocks[:self.N]

        # 更新当前持仓组合
        self.current_portfolio = to_buy

        # 执行交易
        self.order_stock_sell(to_buy)
        self.order_stock_buy(to_buy)

        # 记录投资组合
        self.log_portfolio(to_buy)

        self.if_trade = False

    def get_ranked_factors(self):
        """获取排序后的因子数据"""
        factor_values = []
        stock_codes = []

        for stock in self.all_stocks:
            # 获取基本面数据
            fundamental = self.data_handler.get_fundamental_data(stock, self.current_date)

            # 获取价格预测
            predicted_price = self.data_handler.predict_price(stock, self.current_date)
            if predicted_price is not None:
                # 计算预期收益率
                current_price_data = self.data_handler.get_stock_price(stock, self.current_date, self.current_date)
                if not current_price_data.empty:
                    current_price = current_price_data.iloc[-1]['close']
                    expected_return = (predicted_price - current_price) / current_price
                    # 将预期收益率作为一个额外的因子
                    fundamental['expected_return'] = expected_return

            # 提取需要的因子值
            stock_factors = [fundamental.get(factor, 0) for factor in self.factors]
            factor_values.append(stock_factors)
            stock_codes.append(stock)

        # 转换为NumPy数组
        factor_array = np.array(factor_values)

        # 因子标准化处理
        normalized_factors = self.normalize_factors(factor_array)

        return normalized_factors, stock_codes

    def normalize_factors(self, factors):
        """因子标准化处理"""
        # 简单处理：按列排序
        ranked_factors = np.zeros_like(factors)
        for i in range(factors.shape[1]):
            col = factors[:, i]
            ranked = col.argsort().argsort()  # 获取排名
            ranked_factors[:, i] = ranked
        return ranked_factors

    def order_stock_sell(self, to_buy):
        """卖出不在买入列表中的股票"""
        positions_to_sell = [stock for stock in self.portfolio['positions'] if stock not in to_buy]

        for stock in positions_to_sell:
            # 获取当前价格
            price_data = self.data_handler.get_stock_price(stock, self.current_date, self.current_date)
            if not price_data.empty:
                # 取最后一行（最新价格）
                current_price = price_data.iloc[-1]['close']
                # 卖出全部持仓
                position = self.portfolio['positions'][stock]
                sell_value = position['shares'] * current_price
                self.portfolio['cash'] += sell_value
                # 记录交易
                self.trade_history.append({
                    'date': self.current_date,
                    'stock': stock,
                    'action': 'sell',
                    'shares': position['shares'],
                    'price': current_price,
                    'value': sell_value
                })
                # 记录日志
                profit = (current_price - position['avg_price']) / position['avg_price'] * 100
                self.log_message(
                    f"卖出 {stock}: {position['shares']}股, 价格: {current_price:.2f}, 盈亏: {profit:.2f}%")
                # 移除持仓
                del self.portfolio['positions'][stock]

    def order_stock_buy(self, to_buy):
        """买入目标股票"""
        for stock in to_buy:
            # 获取当前价格
            price_data = self.data_handler.get_stock_price(stock, self.current_date, self.current_date)
            if price_data.empty:
                continue

            current_price = price_data.iloc[-1]['close']

            # 计算可买股数
            shares_to_buy = int(self.every_stock / current_price)
            if shares_to_buy == 0:
                continue

            # 计算购买金额
            buy_value = shares_to_buy * current_price

            # 检查现金是否足够
            if buy_value > self.portfolio['cash']:
                # 现金不足，调整购买数量
                shares_to_buy = int(self.portfolio['cash'] / current_price)
                buy_value = shares_to_buy * current_price
                if shares_to_buy == 0:
                    continue

            # 更新现金
            self.portfolio['cash'] -= buy_value

            # 更新持仓
            if stock in self.portfolio['positions']:
                position = self.portfolio['positions'][stock]
                total_shares = position['shares'] + shares_to_buy
                avg_price = (position['shares'] * position['avg_price'] + buy_value) / total_shares
                self.portfolio['positions'][stock] = {
                    'shares': total_shares,
                    'avg_price': avg_price
                }
            else:
                self.portfolio['positions'][stock] = {
                    'shares': shares_to_buy,
                    'avg_price': current_price
                }

            # 记录交易
            self.trade_history.append({
                'date': self.current_date,
                'stock': stock,
                'action': 'buy',
                'shares': shares_to_buy,
                'price': current_price,
                'value': buy_value
            })

            # 记录日志
            self.log_message(f"买入 {stock}: {shares_to_buy}股, 价格: {current_price:.2f}, 金额: {buy_value:.2f}")

    def update_portfolio_value(self):
        """更新投资组合价值"""
        total_value = self.portfolio['cash']

        for stock, position in self.portfolio['positions'].items():
            # 获取当前价格
            price_data = self.data_handler.get_stock_price(stock, self.current_date, self.current_date)
            if not price_data.empty:
                current_price = price_data.iloc[-1]['close']
                market_value = position['shares'] * current_price
                total_value += market_value

        self.portfolio['value'] = total_value
        return total_value

    def log_portfolio(self, portfolio):
        """记录投资组合"""
        self.log_message(f"当前投资组合 ({len(portfolio)}只股票):")
        for i, stock in enumerate(portfolio, 1):
            price_data = self.data_handler.get_stock_price(stock, self.current_date, self.current_date)
            if not price_data.empty:
                current_price = price_data.iloc[-1]['close']
                predicted_price = self.data_handler.predict_price(stock, self.current_date)
                if predicted_price is not None:
                    expected_return = (predicted_price - current_price) / current_price * 100
                    self.log_message(
                        f"{i}. {stock}: 现价 {current_price:.2f}, 预测价 {predicted_price:.2f}, 预期收益 {expected_return:.2f}%")