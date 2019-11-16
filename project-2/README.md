## Install
```shell script
# create environment
conda env create -f environment.yml

# create folders
mkdir objeect-recongition/{data,output}
```

## Download Data
Image classification uses batch-1 CIFAR-10 dataset for training, which contains 10000 images; And a trimmed test set 
containing 2000 images is used. 

The needed data can be downloaded from [Google Drive](https://drive.google.com/open?id=1TR9EBK-1NlpTUb8qFcEdHhjjzL25w92W)

You may unzip it and move data into their corresponding folders:
```shell script
unzip 4042-data.zip
mv data_batch_1 object-recognition/data/data_batch_1.pkl
mv test_batch_trim object-recognition/data/test_batch_trim.pkl

```
 
## Answers
### Part A: Object Recognition
#### Question 1. 
Train the network by using mini-batch gradient descent learning. Set batch size =128, and
learning rate 𝛼 = 0.001. Images should be scaled.
Train:
```shell script
python plot_image_classifier.py --output=image-classifier-baseline image-classifier-baseline.yaml
```
Plot test accuracies
Plot feature maps:
```shell script
python train-and-plot-feature-maps.py plot --test_num=2
```

#### Question 2.
Using a grid search, find the optimal numbers of feature maps for part (1) at the convolution
layers. Use the test accuracy to determine the optimal number of feature maps.
Change the settings inside `src/grid_search_train_image_classifier.py`, and run:
```shell script
python grid-search-image-classifier-optim-channels.py search
```
Some technique can be apply to speed up.

For visualization,
```shell script
python grid-search-image-classifier-optim-channels.py <path-to-saved-statistic-result>.pkl
```

The optimal model configuration is in `src/configs/'

#### Question 3.
Using the optimal number of filters, we train the classifier using:
1. GD with momentum with 𝛾 = 0.1
    ```shell script
    python train-image-classifier-with-stat.py train --momentum=0.1 --output=image-classifier-momentum
    ```
2. Using RMSProp algorithm for learning
    ```shell script
    python train-image-classifier-with-stat.py train --optimizer=rmsprop --output=image-classifier-rmsprop
    ```
3. Using Adam optimizer for learning
    ```shell script
    python train-image-classifier-with-stat.py train --optimizer=adam --output=image-classifier-adam
    ```
4. Dropout
```shell script
python train-image-classifier-with-stat.py train --output=image-classifier-dropout --config=config/image-classifier-dropout.yaml
````
5. Plot
```shell script
python train-image-classifier-with-stat.py plot output/<your-output-pickle-name-stat>.pkl
```

### Part B: Text Classification
#### Question 1. 
Char-CNN model
Train:
```shell script
cd questions
python q1.py
```

#### Question 2.
Word-CNN model
```shell script
cd questions
python q2.py
```

#### Question 3.
Char-RNN model
```shell script
cd questions
python q3.py
```

#### Question 4.
Word-RNN model
```shell script
cd questions
python q4.py
```

#### Question 5.
change the keep_prob variable value in feed_dict in q1.py, q2.py, q3.py, q4.py and run
```shell script
cd questions
python q1.py
```


#### Question 6.
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

5. Plotting: 
    ```shell script
    python plot_graph.py -file_dir [pickle_file_saved]
    ```
