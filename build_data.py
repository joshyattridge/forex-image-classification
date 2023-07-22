import pandas as pd
import numpy as np
import cv2
import io
from PIL import Image
import os
import multiprocessing

# import all from strategy_builder.py
from strategy_builder import *

train_split = 0.7
test_split = 0.2
validation_split = 0.1

def create_folders(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

create_folders([
    'image_data',
    'image_data/train',
    'image_data/test',
    'image_data/validation',
    'image_data/train/win',
    'image_data/train/loss',
    'image_data/test/win',
    'image_data/test/loss',
])

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

    fig = add_indicators_to_fig(fig, candle_data)

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

    return img

def get_chart_data(file):
    pair = file.split(".")[0]

    path = os.path.join("data", file)
    chart_data = pd.read_csv(path)
    chart_data = add_indicators(chart_data)
    atr_multiplier = 1.5
    chart_data['SLDistance'] = round(TA.ATR(chart_data, 14) * atr_multiplier,5)
    chart_data.drop(['time'], axis=1, inplace=True)
    chart_data = chart_data.iloc[210:].reset_index(drop=True)
    return chart_data, pair

def get_trade_result(chart_data, candle_pos, aim):
    if aim:
        if candle_pos + timeout_limit > len(chart_data):
            return None, None, None
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
        if (hit_short_SL == False and hit_short_TP == True and aim == "short") or (hit_long_SL == False and hit_long_TP == True and aim == "long"):
            return True, window_candle_data, SLDistance
        else:
            return False, window_candle_data, SLDistance
    else:
        return None, None, None

def save_image(img, candle_pos, win, split_group, pair):
    # if the split group is validation then add pair to the path
    win = "win" if win else "loss"
    if split_group == "validation":
        create_folders([
            'image_data/validation/' + pair,
            'image_data/validation/' + pair + '/win',
            'image_data/validation/' + pair + '/loss',
        ])
        path = os.path.join("image_data", split_group, pair, win)
    else:
        path = os.path.join("image_data", split_group, win)

    # save the image
    cv2.imwrite(os.path.join(path, pair + "_" + str(candle_pos) + ".png"), img)
    cv2.imwrite("render.png", img)

def data_building(chart_data, pair):

    # find the posistion for each of the splits
    train_split_pos = int(len(chart_data) * train_split)
    test_split_pos = int(len(chart_data) * test_split) + train_split_pos
    validation_split_pos = int(len(chart_data) * validation_split) + test_split_pos

    for candle_pos in range(candle_window_size, len(chart_data)):

        aim = entry_condition(chart_data, candle_pos)

        win, window_candle_data, SLDistance = get_trade_result(chart_data, candle_pos, aim)

        if win != None:
            rendered_img = render(window_candle_data, aim, SLDistance)

            # work out the split group
            if candle_pos < train_split_pos:
                split_group = "train"
            elif candle_pos < test_split_pos:
                split_group = "test"
            elif candle_pos < validation_split_pos:
                split_group = "validation"
            else:
                return

            save_image(rendered_img, candle_pos, win, split_group, pair)

            pair_progress = round(((candle_pos / len(chart_data))) * 100, 2)

            print(pair + " Progress: " + str(pair_progress) + "%", end="\r")

def process_file(file):
    # build the data
    chart_data, pair = get_chart_data(file)
    data_building(chart_data, pair)

if __name__ == '__main__':

    # # for each file in the data folder
    # for file in os.listdir("data"):
    #     # build the data
    #     chart_data, pair = get_chart_data(file)
    #     data_building(chart_data, pair)

    # create a pool of worker processes
    pool = multiprocessing.Pool()

    # for each file in the data folder, submit a job to the pool
    for file in os.listdir("data"):
        pool.apply_async(process_file, args=(file,))

    # wait for all jobs to complete
    pool.close()
    pool.join()