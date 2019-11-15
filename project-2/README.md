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

