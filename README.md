## My Solution: Neural-PID controller

At first, I tried using a (fairly small) LSTM network to map all given input information to the desired steering outcome using the first 100 time steps from the data with labelled steering commands in a straighforward supervised learning problem. However, although I was able to produce results on par with the simple controller via this technique (so something was working), I determined that since during test time the target and current lat accels are not equal (unlike during train time here), there is a significant training-serving skew which may have led to subpotimal results. Furthermore, since I don't have access to a local GPU, I was training on google colab with paid compute units and I didn't want to spend more time and money training different models/tuning hypers, especially since I didn't expect great results given the training-serving skew.

I could have potentially created a synthetic dataset via small perterbations to produce data points without equal target and current lat accels, but I wanted to try other ideas first.

I'm a graduating senior (only a few days left till I have my degree) and in one of my earliest classes we wrote a PID controller to drive a line-following car. To be honest, I haven't dealt with controls directly very much since that class, but I decided to give it a shot. At first I manually tried various values for the PID coefficients, but quickly realized automation was the right move. At my internship at Intel I implemented a genetic algorithm search for optimal DDR5 memory parameters, so I thought I could try something similar here. So, I implemented a fairly simple GA to search for the PID coefficients. After running this, I tested my final values and was pleased with the results.

With this baseline controller, I wanted to see what I could add to improve it. This reminded me of another time I had tried to correct a Kalman filter via a neural net, so I thought I could try something similar here. To make it easier to train, I took just the three state params, not the target/current lat accels, as input to my LSTM and trained it to predict the difference between the optimal steering command and the PID predicted one. To get the data for this, I modified a few of the given files so I could log my PID controller's output, then I used the labelled steering time steps and subtracted the values to obtain the differences. From here, it was just a matter of training the LSTM. Again, I used google colab to do this and I didn't want to use too much time/money to tune it, so I tried a few things and ended up with an improvement over the PID controller.

To clarify, I use the PID controller I found using my GA search and then ADD ON the prediction of the LSTM network and this is my final steering output. In other words, the LSTM net predicts the error of the PID controller alone and tries to correct it.

If I had more time, I would refine the PID controller search (since my GA is pretty simple right now) and I would try some different architectures for my neural net. Using a transformer-based architecture here might be interesting since I wonder if there are key moments while steering that make the difference between being on track and not for the next few seconds and I wonder if an attention mechanism might use that fact more. In any case, I'm happy with what I have produced given my time/resources and I might want to keep working on this once I have more time after graduation.

Final scores: 5 * (1.855) + 24.819 = 34.096

Note: I was having issues running the new version of eval.py with my LSTM backed PID controller, so I switched back to using the original one. Not sure why, could have spent more time looking into this too.


# Comma Controls Challenge!
![Car](./imgs/car.jpg)

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.


## Geting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual routes with actual car and road states.

```
# download necessary dataset (~0.6G)
bash ./download_dataset.sh

# install required packages
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller simple
```

There are some other scripts to help you get aggregate metrics: 
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller simple

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller simple --baseline_controller open

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. It's inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`) and a steer input (`steer_action`) and predicts the resultant lateral acceleration fo the car.


## Controllers
Your controller should implement an [update function](https://github.com/commaai/controls_challenge/blob/1a25ee200f5466cb7dc1ab0bf6b7d0c67a2481db/controllers.py#L2) that returns the `steer_action`. This controller is then run in-loop, in the simulator to autoregressively predict the car's response.


## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(actual\\_lat\\_accel - target\\_lat\\_accel)^2}{steps} * 100$

- `jerk_cost`: $\dfrac{\Sigma((actual\\_lat\\_accel\_t - actual\\_lat\\_accel\_{t-1}) / \Delta t)^2}{steps - 1} * 100$

It is important to minimize both costs. `total_cost`: $(lataccel\\_cost *5) + jerk\\_cost$

## Submission
Run the following command, and send us a link to your fork of this repo, and the `report.html` this script generates.
```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller simple
```

## Work at comma
Like this sort of stuff? You might want to work at comma!
https://www.comma.ai/jobs
