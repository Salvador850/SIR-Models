import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
import pandas as pd
import sys
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from progress.bar import Bar

# Gathers test data from file
data_file = open(str(sys.argv[1]), 'r').read()
target_list = eval(data_file)
timestamp_list = [i for i in range(len(target_list))]
item_id_list = [0 for i in range(len(target_list))]

full_data = zip(item_id_list, timestamp_list, target_list)
data_frame = pd.DataFrame(full_data, columns=["item_id", "timestamp", "target"])

data_frame['target']  /= 4 * (10**6)

print(sum(data_frame['target'])/len(data_frame['target']))

plt.plot(list(data_frame['target'])[:1000], color="blue")
plt.show()

# Call Pretrained Model
prediction_length = int(sys.argv[3])
pretrained = TimeSeriesPredictor.load(str(sys.argv[2]))

# Initialize lists to save losses to.

timestep = []
zero_shot_loss = []
fine_tuned_loss = []

bar = Bar('Running Simulation', max=int(len(data_frame) / 50 ) - 4)

# Loops over data adding another prediction frame each step, 8
# Note that the models might not get full context length(400) for the first few steps
for i in range(2, int(len(data_frame) / prediction_length)):
    timestep.append((i * prediction_length) + prediction_length)
    filtered_df = data_frame.loc[data_frame['timestamp'] <= i*50]
    filtered_df["timestamp"] *= 10**13

    # Get data up to the timestamp the loop is currently on
    data = TimeSeriesDataFrame.from_data_frame(
        filtered_df,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Fit zero shot model
    train_data, test_data = data.train_test_split(prediction_length)
    predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MSE")
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "Chronos": [
                {"model_path": "bolt_base", "fine_tune": False, "ag_args": {"name_suffix": "ZeroShot"}},
            ]
        },
        time_limit=250, 
        enable_ensemble=False,
        verbosity=1
    )

    # Get predictions for both model and save them to lists
    predictions = pretrained.predict(train_data[-400:])
    zero_shot_predictions = predictor.predict(train_data[-400:])

    
    fine_tuned_frame_loss = (predictions['mean'] - test_data['target'][-50:])**2
    fine_tuned_loss.append(sum(fine_tuned_frame_loss)/len(fine_tuned_frame_loss))
    zero_shot_frame_loss = (zero_shot_predictions['mean'] - test_data['target'][-50:])**2
    zero_shot_loss.append(sum(zero_shot_frame_loss)/len(zero_shot_frame_loss))
    bar.next()

    
bar.finish()

# Print outputs and plot loss for each prediction
print("Overall Zero Shot MSE:", sum(zero_shot_loss)/len(zero_shot_loss), "All Points:", zero_shot_loss)
print("Overall Fine Tuned MSE:", sum(fine_tuned_loss)/len(fine_tuned_loss), "All Points:", fine_tuned_loss)

plt.plot(timestep, zero_shot_loss, label="Zero Shot Loss")
plt.plot(timestep, fine_tuned_loss, label="Fine Tuned Loss")

plt.xlabel('Time Steps Predicted')
plt.ylabel('Loss(MSE)')
plt.title('Zero Chronos Loss for each Predictions')

plt.legend()
plt.show()




