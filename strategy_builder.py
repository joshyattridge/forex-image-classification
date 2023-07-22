import os
import plotly.graph_objects as go
from finta import TA

candle_window_size = 200
timeout_limit = 100
risk_to_reward = 2

def add_indicators(chart_data):
    chart_data['BB_upper'] = TA.BBANDS(chart_data, 20)['BB_UPPER']
    chart_data['BB_lower'] = TA.BBANDS(chart_data, 20)['BB_LOWER']

    # add 200 sma
    chart_data['SMA_200'] = TA.SMA(chart_data, 200)

    return chart_data

def add_indicators_to_fig(fig, candle_data):
    # add the bollinger bands
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(candle_window_size)],
            y=candle_data["BB_upper"],
            line=dict(color="gray", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(candle_window_size)],
            y=candle_data["BB_lower"],
            line=dict(color="gray", width=1),
        )
    )

    # add the 200 sma
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(candle_window_size)],
            y=candle_data["SMA_200"],
            line=dict(color="purple", width=1),
        )
    )

    return fig

def entry_condition(chart_data, candle_pos):
    if chart_data['high'].iloc[candle_pos] > chart_data['BB_upper'].iloc[candle_pos]:
        return "short"
    elif chart_data['low'].iloc[candle_pos] < chart_data['BB_lower'].iloc[candle_pos]:
        return "long"
    else:
        return False
