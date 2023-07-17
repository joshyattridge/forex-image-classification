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
import shutil

# create a folder called image_data
if not os.path.exists('image_data'):
    os.makedirs('image_data')
# create a folder for each trend type
if not os.path.exists('image_data/win'):
    os.makedirs('image_data/win')
if not os.path.exists('image_data/loss'):
    os.makedirs('image_data/loss')

candle_window_size = 200
timeout_limit = 100
# stoploss = 0.005
risk_to_reward = 2

def render(candle_data,trend,SLDistance):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            open=candle_data["open"],
            high=candle_data["high"],
            low=candle_data["low"],
            close=candle_data["close"]
        )
    )

    # the candles are too thick for the image
    fig.update_traces(
        line=dict(width=1),
        selector=dict(type="candlestick")
    )

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

    # make the background white
    fig.update_layout(
        plot_bgcolor="white",
    )

    # remove the legend
    fig.update_layout(showlegend=False)

    SL = SLDistance
    if trend == "long":
        SL = -SLDistance

    fig.add_shape(
        dict(
            type="line",
            x0=0,
            y0=candle_data["close"].iloc[-1] + SL,
            x1=candle_window_size,
            y1=candle_data["close"].iloc[-1] + SL,
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
    fig.update_layout(width=len(candle_data), height=100)

    # convert the plotly figure to a numpy array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    img = np.asarray(img)
    # convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # if the colors in the image are not white then make them black
    # img[img < 255] = 0

    return img

def save_image(img, win):
    # save the image to the correct folder
    if win:
        path = 'image_data/win'
    else:
        path = 'image_data/loss'

    # save the image
    cv2.imwrite(os.path.join(path, str(len(os.listdir(path))) + '.png'), img)
    cv2.imwrite("render.png", img)

def data_building(file):

    chart_data = pd.read_csv(os.path.join(path, file))

    chart_data['BB_upper'] = TA.BBANDS(chart_data, 20)['BB_UPPER']
    chart_data['BB_lower'] = TA.BBANDS(chart_data, 20)['BB_LOWER']

    # add 200 sma
    chart_data['SMA_200'] = TA.SMA(chart_data, 200)

    atr_multiplier = 1.5
    chart_data['SLDistance'] = round(TA.ATR(chart_data, 14) * atr_multiplier,5)
    chart_data.drop(['time'], axis=1, inplace=True)
    chart_data = chart_data.iloc[210:].reset_index(drop=True)

    for candle_pos in range(candle_window_size, len(chart_data)):

        # if the candles high is above the upper bollinger band then it is a short trend
        # if the candles low is below the lower bollinger band then it is a long trend
        # otherwise return
        if chart_data['high'].iloc[candle_pos] > chart_data['BB_upper'].iloc[candle_pos]:
            aim = "short"
        elif chart_data['low'].iloc[candle_pos] < chart_data['BB_lower'].iloc[candle_pos]:
            aim = "long"
        else:
            continue

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
        
        win = False
        if hit_short_SL == False and hit_short_TP == True and aim == "short":
            # short trend
            win = True
        if hit_long_SL == False and hit_long_TP == True and aim == "long":
            # long trend
            win = True

        rendered_img = render(window_candle_data, aim, SLDistance)
        save_image(rendered_img, win)

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

# wait for all the threads to finish
for thread in data_building_threads:
    thread.join()

# without threading
# for file in files:
#     data_building(file)

# split the data into train and test and put them in sub folders
for folder in ['train/win', 'train/loss', 'test/win', 'test/loss']:
    os.makedirs(os.path.join(cwd, 'image_data', folder))

for folder in ['win', 'loss']:
    files = os.listdir(os.path.join(cwd, 'image_data', folder))
    for i in range(len(files)):
        if i < len(files) * 0.8:
            # 80% train
            shutil.move(os.path.join(cwd, 'image_data', folder, files[i]), os.path.join(cwd, 'image_data', 'train', folder, files[i]))
        else:
            # 20% test
            shutil.move(os.path.join(cwd, 'image_data', folder, files[i]), os.path.join(cwd, 'image_data', 'test', folder, files[i]))
    
# delete the win and loss folders
shutil.rmtree(os.path.join(cwd, 'image_data', 'win'))
shutil.rmtree(os.path.join(cwd, 'image_data', 'loss'))

print("Done")
