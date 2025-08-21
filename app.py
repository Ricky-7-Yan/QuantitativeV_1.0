import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 自定义CSS样式
def load_css():
    st.markdown("""
    <style>
    /* 自定义样式 */
    .stButton button {
        width: 100%;
    }

    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }

    .css-1v0mbdj {
        padding: 10px;
    }

    /* 卡片样式 */
    .card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }

    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* 副标题样式 */
    .sub-title {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }

    /* 信息框样式 */
    .info-box {
        background-color: #e1f5fe;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #0288d1;
    }

    /* 警告框样式 */
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #ff9800;
    }

    /* 错误框样式 */
    .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #f44336;
    }

    /* 成功框样式 */
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)


# 页面设置
st.set_page_config(
    page_title="多因子策略交易平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入其他模块
try:
    from backtest import BacktestEngine
    import config
    import data_utils
except ImportError as e:
    st.error(f"导入模块错误: {e}")
    st.info("请确保所有必要文件都存在: backtest.py, config.py, data_utils.py")
    st.stop()

# 初始化会话状态
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'backtest_done' not in st.session_state:
    st.session_state.backtest_done = False
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = config.STOCK_POOL[0] if hasattr(config,
                                                                      'STOCK_POOL') and config.STOCK_POOL else "600519.SS"
if 'date_range' not in st.session_state:
    st.session_state.date_range = {
        'start': config.STRATEGY_CONFIG['start_date'] if hasattr(config, 'STRATEGY_CONFIG') else '2020-01-01',
        'end': config.STRATEGY_CONFIG['end_date'] if hasattr(config, 'STRATEGY_CONFIG') else '2023-12-31'
    }

# 加载CSS
load_css()

# 标题
st.markdown('<h1 class="main-title">📈 多因子策略交易平台</h1>', unsafe_allow_html=True)
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("策略配置")

    # 确保配置存在
    if not hasattr(config, 'STRATEGY_CONFIG'):
        config.STRATEGY_CONFIG = {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'capital_base': 1000000,
            'rebalance_freq': 15,
            'sample_period': 63,
            'hold_num': 10,
            'factors': ['market_cap', 'roe', 'pe_ratio'],
            'factor_weights': [1.0, -1.0, -1.0],
            'index': '000300.SS'
        }

    # 确保股票池存在
    if not hasattr(config, 'STOCK_POOL'):
        config.STOCK_POOL = [
            '600519.SS', '000858.SZ', '601318.SS', '600036.SS', '000333.SZ',
            '600900.SS', '601888.SS', '600276.SS', '600887.SS', '601166.SS',
            '601668.SS', '601328.SS', '601398.SS', '601288.SS', '601988.SS'
        ]

    # 回测参数
    start_date = st.date_input("开始日期", datetime.strptime(config.STRATEGY_CONFIG['start_date'], '%Y-%m-%d'))
    end_date = st.date_input("结束日期", datetime.strptime(config.STRATEGY_CONFIG['end_date'], '%Y-%m-%d'))
    capital_base = st.number_input("初始资金", value=config.STRATEGY_CONFIG['capital_base'], step=100000)
    rebalance_freq = st.slider("调仓频率(天)", 5, 30, config.STRATEGY_CONFIG['rebalance_freq'])
    hold_num = st.slider("持仓数量", 5, 20, config.STRATEGY_CONFIG['hold_num'])

    # 因子权重
    st.subheader("因子权重")
    # 修复：确保默认值是浮点数而不是列表
    market_cap_weight = st.slider("市值因子权重", -1.0, 1.0, float(config.STRATEGY_CONFIG['factor_weights'][0]), 0.1)
    roe_weight = st.slider("ROE因子权重", -1.0, 1.0, float(config.STRATEGY_CONFIG['factor_weights'][1]), 0.1)
    pe_weight = st.slider("PE因子权重", -1.0, 1.0, float(config.STRATEGY_CONFIG['factor_weights'][2]), 0.1)

    # 更新配置
    config.STRATEGY_CONFIG['start_date'] = start_date.strftime('%Y-%m-%d')
    config.STRATEGY_CONFIG['end_date'] = end_date.strftime('%Y-%m-%d')
    config.STRATEGY_CONFIG['capital_base'] = capital_base
    config.STRATEGY_CONFIG['rebalance_freq'] = rebalance_freq
    config.STRATEGY_CONFIG['hold_num'] = hold_num
    config.STRATEGY_CONFIG['factor_weights'] = [market_cap_weight, roe_weight, pe_weight]

    # 运行回测按钮
    if st.button("运行回测", type="primary"):
        with st.spinner("回测中，请稍候..."):
            try:
                st.session_state.engine = BacktestEngine(config.STRATEGY_CONFIG)
                st.session_state.engine.run()
                st.session_state.backtest_done = True
                st.success("回测完成!")
            except Exception as e:
                st.error(f"回测过程中出现错误: {e}")

    st.markdown("---")

    # 股票选择
    st.subheader("股票分析")
    selected_stock = st.selectbox("选择股票", config.STOCK_POOL)
    st.session_state.selected_stock = selected_stock

    # K线图日期范围
    kline_start = st.date_input("K线开始日期", datetime.strptime(config.STRATEGY_CONFIG['start_date'], '%Y-%m-%d'))
    kline_end = st.date_input("K线结束日期", datetime.strptime(config.STRATEGY_CONFIG['end_date'], '%Y-%m-%d'))
    st.session_state.date_range = {
        'start': kline_start.strftime('%Y-%m-%d'),
        'end': kline_end.strftime('%Y-%m-%d')
    }

# 主内容区域
tab1, tab2, tab3, tab4 = st.tabs(["📊 回测结果", "📈 股票分析", "💼 投资组合", "📋 交易日志"])

with tab1:
    if st.session_state.backtest_done:
        # 回测结果分析
        try:
            results = st.session_state.engine.analyze_results()
            results_df = results['results']

            # 关键指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最终资产", f"{results['final_value']:,.2f}元")
            with col2:
                st.metric("年化收益率", f"{results['annual_return'] * 100:.2f}%")
            with col3:
                st.metric("最大回撤", f"{results['max_drawdown'] * 100:.2f}%")
            with col4:
                st.metric("夏普比率", f"{results['sharpe_ratio']:.2f}")

            # 净值曲线
            st.subheader("净值曲线")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['portfolio_value'],
                                     mode='lines', name='投资组合净值', line=dict(color='blue')))
            fig.add_hline(y=config.STRATEGY_CONFIG['capital_base'], line_dash="dash",
                          line_color="red", annotation_text="初始资金")
            fig.update_layout(height=400, xaxis_title="日期", yaxis_title="净值")
            st.plotly_chart(fig, use_container_width=True)

            # 回撤曲线
            st.subheader("回撤曲线")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=results_df.index, y=results_df['drawdown'] * 100,
                                      mode='lines', name='回撤', line=dict(color='red')))
            fig2.update_layout(height=300, xaxis_title="日期", yaxis_title="回撤(%)")
            st.plotly_chart(fig2, use_container_width=True)

            # 收益率分布
            st.subheader("收益率分布")
            returns = results_df['return'].dropna()
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=returns * 100, nbinsx=50, name='收益率分布'))
            fig3.update_layout(height=300, xaxis_title="日收益率(%)", yaxis_title="频次")
            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"分析回测结果时出错: {e}")
    else:
        st.info("请先运行回测以查看结果")

with tab2:
    st.subheader(f"{st.session_state.selected_stock} 分析")

    # 获取股票数据
    try:
        data_handler = data_utils.DataHandler(st.session_state.date_range['start'], st.session_state.date_range['end'])
        stock_data = data_handler.get_stock_price(
            st.session_state.selected_stock,
            st.session_state.date_range['start'],
            st.session_state.date_range['end']
        )

        if not stock_data.empty:
            # K线图
            st.subheader("K线图")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.1, subplot_titles=('K线图', '成交量'),
                                row_width=[0.2, 0.7])

            # 添加K线
            fig.add_trace(go.Candlestick(x=stock_data.index,
                                         open=stock_data['open'],
                                         high=stock_data['high'],
                                         low=stock_data['low'],
                                         close=stock_data['close'],
                                         name='K线'), row=1, col=1)

            # 添加移动平均线
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['ma5'],
                                     mode='lines', name='MA5', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['ma20'],
                                     mode='lines', name='MA20', line=dict(color='purple')), row=1, col=1)

            # 添加成交量
            colors = ['red' if row['close'] >= row['open'] else 'green' for _, row in stock_data.iterrows()]
            fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['volume'],
                                 name='成交量', marker_color=colors), row=2, col=1)

            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # 技术指标
            st.subheader("技术指标")
            col1, col2, col3 = st.columns(3)

            with col1:
                current_price = stock_data.iloc[-1]['close']
                prev_price = stock_data.iloc[-2]['close'] if len(stock_data) > 1 else current_price
                change = ((current_price - prev_price) / prev_price) * 100
                st.metric("当前价格", f"{current_price:.2f}", f"{change:.2f}%")

            with col2:
                ma5 = stock_data.iloc[-1]['ma5']
                ma20 = stock_data.iloc[-1]['ma20']
                st.metric("MA5/MA20", f"{ma5:.2f}/{ma20:.2f}")

            with col3:
                rsi = stock_data.iloc[-1]['rsi'] if not pd.isna(stock_data.iloc[-1]['rsi']) else "N/A"
                st.metric("RSI", f"{rsi:.2f}" if isinstance(rsi, (int, float)) else rsi)

            # 价格预测
            st.subheader("价格预测")
            predicted_price = data_handler.predict_price(st.session_state.selected_stock,
                                                         st.session_state.date_range['end'])
            if predicted_price is not None:
                expected_return = ((predicted_price - current_price) / current_price) * 100
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("预测价格", f"{predicted_price:.2f}")
                with col2:
                    st.metric("预期收益率", f"{expected_return:.2f}%")

                # 买卖建议
                if expected_return > 5:
                    st.success("强烈建议买入")
                elif expected_return > 0:
                    st.info("建议买入")
                elif expected_return > -5:
                    st.warning("建议观望")
                else:
                    st.error("建议卖出")
            else:
                st.warning("无法生成价格预测")
        else:
            st.error("无法获取股票数据")
    except Exception as e:
        st.error(f"获取股票数据时出错: {e}")

with tab3:
    if st.session_state.backtest_done:
        st.subheader("当前投资组合")

        # 获取持仓信息
        try:
            portfolio = st.session_state.engine.strategy.portfolio
            trade_history = st.session_state.engine.get_trade_history()

            # 持仓股票
            if portfolio['positions']:
                positions_df = pd.DataFrame.from_dict(portfolio['positions'], orient='index')
                positions_df['stock'] = positions_df.index
                positions_df = positions_df.reset_index(drop=True)

                # 计算当前市值
                current_values = []
                for stock in positions_df['stock']:
                    price_data = data_handler.get_stock_price(stock, st.session_state.date_range['end'],
                                                              st.session_state.date_range['end'])
                    if not price_data.empty:
                        current_price = price_data.iloc[-1]['close']
                        shares = positions_df[positions_df['stock'] == stock]['shares'].values[0]
                        current_value = shares * current_price
                        current_values.append(current_value)
                    else:
                        current_values.append(0)

                positions_df['current_value'] = current_values
                positions_df['weight'] = positions_df['current_value'] / portfolio['value'] * 100

                # 显示持仓表格
                st.dataframe(positions_df[['stock', 'shares', 'avg_price', 'current_value', 'weight']].rename(
                    columns={'stock': '股票', 'shares': '股数', 'avg_price': '平均成本',
                             'current_value': '当前市值', 'weight': '权重(%)'}
                ), use_container_width=True)

                # 持仓饼图
                st.subheader("持仓分布")
                fig = go.Figure(data=[go.Pie(labels=positions_df['stock'], values=positions_df['current_value'])])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("当前没有持仓")

            # 资产分配
            st.subheader("资产分配")
            fig = go.Figure(data=[
                go.Pie(labels=['股票', '现金'], values=[portfolio['value'] - portfolio['cash'], portfolio['cash']])])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"获取投资组合信息时出错: {e}")
    else:
        st.info("请先运行回测以查看投资组合")

with tab4:
    if st.session_state.backtest_done:
        st.subheader("交易日志")

        # 显示日志
        try:
            logs = st.session_state.engine.get_logs()
            log_text = "\n".join(logs)
            st.text_area("日志内容", log_text, height=400)

            # 交易历史
            st.subheader("交易历史")
            trade_history = st.session_state.engine.get_trade_history()
            if not trade_history.empty:
                st.dataframe(trade_history, use_container_width=True)

                # 交易统计
                st.subheader("交易统计")
                buy_trades = trade_history[trade_history['action'] == 'buy']
                sell_trades = trade_history[trade_history['action'] == 'sell']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总交易次数", len(trade_history))
                with col2:
                    st.metric("买入次数", len(buy_trades))
                with col3:
                    st.metric("卖出次数", len(sell_trades))
            else:
                st.info("没有交易记录")
        except Exception as e:
            st.error(f"获取交易日志时出错: {e}")
    else:
        st.info("请先运行回测以查看日志")

# 页脚
st.markdown("---")
st.markdown("多因子策略交易平台 © 2023 | 基于Streamlit构建")