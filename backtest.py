import pandas as pd
import strategy
import data_utils
import config
import numpy as np
from tqdm import tqdm


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.data_handler = data_utils.DataHandler(config['start_date'], config['end_date'])
        self.strategy = strategy.MultiFactorStrategy(self.data_handler)
        self.results = []

    def run(self):
        """运行回测"""
        # 初始化策略
        self.strategy.initialize()

        # 获取所有交易日
        trading_dates = self.data_handler.get_trading_dates(self.config['start_date'], self.config['end_date'])
        print(f"总交易日数: {len(trading_dates)}")

        # 使用tqdm显示进度条
        for i in tqdm(range(len(trading_dates)), desc="回测进度"):
            date = trading_dates[i]
            # 开盘前处理
            self.strategy.before_trading_start(date)

            # 更新组合价值
            self.strategy.update_portfolio_value()

            # 记录每日净值
            self.record_daily_result(date)

            # 执行交易
            self.strategy.handle_data()

    def record_daily_result(self, date):
        """记录每日结果"""
        daily_result = {
            'date': date,
            'portfolio_value': self.strategy.portfolio['value'],
            'cash': self.strategy.portfolio['cash'],
            'positions': len(self.strategy.portfolio['positions'])
        }
        self.results.append(daily_result)

    def get_results(self):
        """获取回测结果"""
        return pd.DataFrame(self.results).set_index('date')

    def get_trade_history(self):
        """获取交易历史"""
        return pd.DataFrame(self.strategy.trade_history)

    def get_logs(self):
        """获取日志"""
        return self.strategy.logs

    def analyze_results(self):
        """分析回测结果"""
        results_df = self.get_results()

        if len(results_df) == 0:
            return {
                'final_value': self.config['capital_base'],
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'results': results_df
            }

        # 计算收益率
        results_df['return'] = results_df['portfolio_value'].pct_change()
        results_df['cum_return'] = (1 + results_df['return']).cumprod() - 1

        # 计算年化收益率
        total_days = len(results_df)
        annual_return = (results_df['portfolio_value'].iloc[-1] / self.config['capital_base']) ** (252 / total_days) - 1

        # 计算最大回撤
        results_df['peak'] = results_df['portfolio_value'].cummax()
        results_df['drawdown'] = (results_df['portfolio_value'] - results_df['peak']) / results_df['peak']
        max_drawdown = results_df['drawdown'].min()

        # 计算夏普比率
        daily_risk_free = 0.0001  # 假设无风险利率
        if results_df['return'].std() > 0:
            sharpe_ratio = (results_df['return'].mean() - daily_risk_free) / results_df['return'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        return {
            'final_value': results_df['portfolio_value'].iloc[-1],
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'results': results_df
        }

    def get_current_portfolio(self):
        """获取当前投资组合"""
        return self.strategy.current_portfolio