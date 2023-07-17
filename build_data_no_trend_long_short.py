# import libraries that are needed
import pandas as pd
import numpy as np
import os
import cv2
import io
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
from finta import TA
import threading

# create a folder called image_data
if not os.path.exists('image_data'):
    os.makedirs('image_data')
# create a folder for each trend type
if not os.path.exists('image_data/no_trend'):
    os.makedirs('image_data/no_trend')
if not os.path.exists('image_data/short_trend'):
    os.makedirs('image_data/short_trend')
if not os.path.exists('image_data/long_trend'):
    os.makedirs('image_data/long_trend')

candle_window_size = 200
timeout_limit = 100
# stoploss = 0.005
risk_to_reward = 2

def render(candle_data,SLDistance):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            open=candle_data["open"],
            high=candle_data["high"],
            low=candle_data["low"],
            close=candle_data["close"], 
        )
    )

    # the candles are too thick for the image
    fig.update_traces(
        line=dict(width=1),
        selector=dict(type="candlestick")
    )

    # make the background white
    fig.update_layout(
        plot_bgcolor="white",
    )

    # remove the legend
    fig.update_layout(showlegend=False)

    # add the stoploss lines
    fig.add_shape(
        dict(
            type="line",
            x0=candle_window_size-1,
            y0=candle_data["close"].iloc[-1] + SLDistance,
            x1=candle_window_size,
            y1=candle_data["close"].iloc[-1] + SLDistance,
            line=dict(color="black", width=1),
        )
    )
    fig.add_shape(
        dict(
            type="line",
            x0=candle_window_size-1,
            y0=candle_data["close"].iloc[-1] - SLDistance,
            x1=candle_window_size,
            y1=candle_data["close"].iloc[-1] - SLDistance,
            line=dict(color="black", width=1),
        )
    )

    # the graph isnt going up to the edge of the image so make it go to the edge
    fig.update_xaxes(range=[0, candle_window_size])

    # remove small graph at the bottom
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=False,
            ),
        )
    )
    # remove white space around the graph
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    # remove the x and y axis
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # lower the resolution of the image to 420p
    fig.update_layout(width=len(candle_data), height=50)

    # convert the plotly figure to a numpy array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    img = np.asarray(img)
    # convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # if the colors in the image are not white then make them black
    img[img < 255] = 0

    return img

def save_image(img, trend):
    # save the image to the correct folder
    if trend == 0:
        path = os.path.join('image_data', 'no_trend')
    elif trend == 1:
        path = os.path.join('image_data', 'short_trend')
    elif trend == 2:
        path = os.path.join('image_data', 'long_trend')
    else:
        return
    # save the image
    cv2.imwrite(os.path.join(path, str(len(os.listdir(path))) + '.png'), img)

def data_building(file):
    chart_data = pd.read_csv(os.path.join(path, file))
    atr_multiplier = 1.5
    chart_data['SLDistance'] = round(TA.ATR(chart_data, 14) * atr_multiplier,5)
    chart_data.drop(['time'], axis=1, inplace=True)
    chart_data = chart_data.iloc[500:].reset_index(drop=True)

    for candle_pos in range(candle_window_size, len(chart_data)):

        if candle_pos + timeout_limit > len(chart_data):
            break

        window_candle_data = chart_data[candle_pos-candle_window_size:candle_pos]
        starting_price = window_candle_data['close'].iloc[-1]
        SLDistance = window_candle_data['SLDistance'].iloc[-1]
        hit_short_SL = False
        hit_short_TP = False
        hit_long_SL = False
        hit_long_TP = False
        for i in range(candle_pos+1, candle_pos + timeout_limit):
            if chart_data['high'].iloc[i] > starting_price + SLDistance:
                hit_short_SL = True
            if chart_data['low'].iloc[i] < starting_price - (SLDistance * risk_to_reward):
                hit_short_TP = True
                break
            if chart_data['low'].iloc[i] < starting_price - SLDistance:
                hit_long_SL = True
            if chart_data['high'].iloc[i] > starting_price + (SLDistance * risk_to_reward):
                hit_long_TP = True
                break
        
        trend = 0
        if hit_short_SL == False and hit_short_TP == True:
            # short trend
            trend = 1
        if hit_long_SL == False and hit_long_TP == True:
            # long trend
            trend = 2

        rendered_img = render(window_candle_data, SLDistance)
        save_image(rendered_img, trend)

        progress = round(((candle_pos / len(chart_data))) * 100, 2)
        print(" Progress: " + str(progress) + "%", end="\r")

# for each file in the data folder
cwd = os.getcwd()
path = os.path.join(cwd, 'data')
files = os.listdir(path)

data_building_threads = []
for file in files:
    data_building_threads.append(threading.Thread(target=data_building, args=(file,)))
    data_building_threads[-1].start()