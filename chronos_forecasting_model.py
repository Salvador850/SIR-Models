from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from matplotlib import pyplot as plt
from progress.bar import Bar
import pandas as pd
import sys
import torch
import torch.nn as nn

# Gather Data 
data_file = open(str(sys.argv[1]), 'r').read()
data_lists = data_file.split(']]')

data_lists =  '[' + ']], '.join(data_lists)[:-2].replace('None', '') + ']'
data_lists = eval(data_lists)
data_frame = pd.DataFrame(data_lists[0], columns=["item_id", "timestamp"] +[str(i) for i in range(len(data_lists[0][0]) - 3)] + ["target"])

bar = Bar('Gathering Data', max=100)
bar.next()
for data in data_lists[1:]:
    for data_point in data:
        data_frame.loc[len(data_frame)] = data_point
    bar.next()

bar.finish()

# Normalize the dfata, convert to time series data frame
data_frame["timestamp"] *= 10**13
data_frame['target']  /= 30
data_frame = data_frame.drop([str(i) for i in range(len(data_lists[0][0]) - 3)], axis=1)

prediction_length = 50

data_frame = data_frame.loc[data_frame['item_id'] <= 50]
full_data = TimeSeriesDataFrame.from_data_frame(
        data_frame,
        id_column="item_id",
        timestamp_column="timestamp"
    )

# If fine tuned is true, run training runs for 1, 2, 3 ... N time series.
# Print losses for all time series up to that point, graph loss over all time
# series at the end.
if eval(sys.argv[2]) == True:
    loss = []
    for i in range(1, sys.argv[3] + 1):
        filtered_df = data_frame.loc[data_frame['item_id'] <= i]
        
        data = TimeSeriesDataFrame.from_data_frame(
            filtered_df,
            id_column="item_id",
            timestamp_column="timestamp"
        )

        trash_train_data, test_data = full_data.train_test_split(prediction_length)
        train_data, trash_test_data = data.train_test_split(prediction_length)
        predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MSE")
        predictor.fit(
            train_data=train_data,
            hyperparameters={
                "Chronos": [
                    {"model_path": "bolt_base", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
                ]
            },
            time_limit=150, 
            enable_ensemble=False,
        )

        loss.append(predictor.evaluate(test_data)['MSE'])
        print("Loss(Fine Tuned Model):", loss)
        
    plt.plot([i for i in range(50)], loss, label="Loss(Same for general and specific fine tuning)")
    plt.xlabel('Time Series Trained')
    plt.ylabel('Loss(MSE)')
    plt.title('Chronos Loss for Amount of Data Trained')
    
    plt.legend()
    plt.show()
# If fine tuned is false run zero shot, print loss
else:

    train_data, test_data = full_ data.train_test_split(prediction_length)
    predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MSE")
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "Chronos": [
                {"model_path": "bolt_base", "fine_tune": False, "ag_args": {"name_suffix": "FineTuned"}},
            ]
        },
        time_limit=250, 
        enable_ensemble=False,
    )
    
    zero_shot_loss = predictor.evaluate(test_data)['MSE']
    print("Loss(Zero Shot Model):", zero_shot_loss)

    predictions = predictor.predict(train_data)
    predictor.plot(
        data=data,
        predictions=predictions,
        item_ids=data.item_ids[:6],
        max_history_length=200,
    );

    plt.show()

