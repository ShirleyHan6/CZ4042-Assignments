## Part A: Object Recognition
### Question 1. 
Train the network by using mini-batch gradient descent learning. Set batch size =128, and
learning rate 𝛼 = 0.001. Images should be scaled.
Train:
```shell script
python plot_image_classifier.py --output=image-classifier-baseline configs/image_classifier_baseline.yaml
```
Plot:
```python
from configs import OUTPUT_DIR
from helper.utils import plot_train_and_test

plot_train_and_test(OUTPUT_DIR / 'image-classifier-stat-baseline.pkl')
```
Plot feature maps:
```shell script
python plot_feature_maps.py
```

### Question 2.
Using a grid search, find the optimal numbers of feature maps for part (1) at the convolution
layers. Use the test accuracy to determine the optimal number of feature maps.
Change the settings inside `src/grid_search_train_image_classifier.py`, and run:
```shell script
python grid_search_train_image_classifier.py
```
Some technique can be apply to speed up.

For visualization,
```python
from grid_search_train_image_classifier import plot
plot()
```

The optimal model configuration is in `src/configs/'

### Question 3.
Using the optimal number of filters, we train the classifier using:
1. GD with momentum with 𝛾 = 0.1
    ```shell script
    python plot_image_classifier.py --momentum=0.1 configs/image_classifier_best.yaml
    ```
2. Using RMSProp algorithm for learning
    ```shell script
    python plot_image_classifier.py --optimizer=rmsprop configs/image_classifier_best.yaml
    ```
3. Using Adam optimizer for learning
    ```shell script
    python plot_image_classifier.py --optimizer=adam configs/image_classifier_best.yaml
    ```

## Part B: Text Classification
Make a data directory and put train_medium.csv and test_medium.csv into the folder
```shell script
mkdir data
```
Make an output directory
```shell script
mkdir output
```
### Question 1. 
Char-CNN model
Train:
```shell script
cd questions
python q1.py
```

### Question 2.
Word-CNN model
```shell script
cd questions
python q2.py
```

### Question 3.
Char-RNN model
```shell script
cd questions
python q3.py
```

### Question 4.
Word-RNN model
```shell script
cd questions
python q4.py
```

### Question 5.
change the keep_prob variable value in feed_dict in q1.py, q2.py, q3.py, q4.py and run
```shell script
cd questions
python q1.py
```


### Question 6.
1. Character-level: RNN, LSTM and double-gru: change the model_type variable value in q6_char.py and run
```shell script
cd questions
python q6_char.py 
```

2. Word-level: RNN, LSTM and double-gru: change the model_type variable value in q6_word.py and run
```shell script
cd questions
python q6_word.py 
```

3. Char-level gradient clipping
```shell script
cd questions
python q6_word_gru_clip.py
```

4. Word-level gradient clipping
```shell script
cd questions
python q6_char_gru_clip.py
```

Plotting: 
```shell script
python plot_graph.py -file_dir [pickle_file_saved]
```

