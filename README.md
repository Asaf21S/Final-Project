# FAsT-Match
FAsT-Match (Fast Affine Template Matching) is an algorithm designed by Simon Korman, Daniel Reichman, Gilad Tsur and Shai Avidan [(source)](https://www.cs.haifa.ac.il/~skorman/FastMatch/index.html) to search a fixed template inside an image, using the B&B technique.  
This is a Python implementation of the FAsT-Match algorithm with threshold learning option.

## The Algorithm
Branch-and-Bound (B&B) is a general technique for accelerating brute-force search in large domains. It is used when a global optimality is desirable, while more efficient optimization methods (like gradient descent) are irrelevant (e.g., due to the minimized function being nonconvex). A particular example of interest, very common in computer vision, is that of template matching (or image alignment), where one image (the template) is searched in another. This is useful, for example, in applications like stitching of panoramas, object localization and tracking.

The template which the FAsT-Match algorithm is trying to find, can be found under an arbitrary affine transformation, therefore the number of transformations to consider is huge.

On each level of the algorithm, the alogorithem checks many transformations (also called configurations) which are spread along the search space. Each configuration is characterized by the photometric distance between the template and the place on the image where this configuration transforms the template into, on sampled points. At the end of each level, we are left with an array of such distances, and we need to decide around which configurations to expand. Therefore, we use a threshold (the higher bound) on those distances’ values, computed as:

**threshold=minimum (called also best)  distance+f(δ)**

The following figure demonstrates how B&B works in FAsT-Match:

![tree_graphic](https://user-images.githubusercontent.com/87817221/185792647-5a916608-b29a-4fb4-9818-f9a3f7b0c74a.png)

The configurations in red were not close enough to expand in the next level (distance>threshold), while the configurations in blue, passed the threshold and therefore were chosen to be expanded in the next level of the algorithm (distance<threshold).

The f(δ) part of the bound is a linear combination of the parameter δ of the algorithm, which specifies the grid resolution of the search space, and is lowered by the same factor each level. This linear combination has been computed manually through trial and error and it uses only one parameter – δ (delta). Because of those reasons, it is not always optimal.

## Learning the Threshold
To tackle the previous issue, this project contains a Neural-Network model implemented in the FAsT-Match algorithm python code and improves the decision process of the abovementioned higher bound, resulting in run-time and accuracy improvement of the FAsT-Match algorithm.

Now, we will consider the threashold to be:

**threshold=minimum distance+f(set of features)**

While the f(set of features) part is now taken from the Neural-Network model, and it isn’t only using a single parameter but a set of features (which includes delta).

The value which our model returns, ideally should be higher than the distance of the nearest configuration in the level, the one which on expansion will eventually get us to the real configuration of the template – the ground truth (green circle in the figure above).

The model used is a Multi-Layer Perceptron for the Regression model, which is a simple Neural-Network consisting of some fully-connected linear layers.

### The final model configuration:

![model_scheme](https://user-images.githubusercontent.com/87817221/185792912-9efb99a2-dd8c-449e-acfa-2b0eb7be0d48.png)

The target value that was used is:
__min_distance+factor*(ground_truth-min_distance)__

By running on various target values to compare the results of each of them to the version of the algorithm without any such model, the optimal factor can be chosen.

We compared the speed results by the average time per run, and the accuracy results by the Jaccard Index (union over bound) as well as the average distance between the corners of the found template to the ground truth one.

On the final round, we measured the results of 15 models with factor in the range 0.15-0.6 and got the following results on 400 test runs:

![model_comperasion](https://user-images.githubusercontent.com/87817221/185971472-d4d5d8ab-f0d5-45a4-ae34-c47079d73bd7.png)

We can clearly notice the trade-off between accuracy and run-time. For instance, models with factor in range 1.5 to 2.1 got corner distance values higher and Jaccard index values lower than the rest of the models, while models with higher factor values got good accuracy results but it took them more seconds to complete a run.

We can also notice by comparing to the first bar which represents the previous method (without model), that our models can barely improve the accuracy, but they can improve the speed of the algorithm, mainly the worst cases as indicated by the average value of the bar in the Time graph. This worst-case improvement is demonstrated in the following graph which shows the time results on 400 runs (on each model), sorted by the time values got on the runs without any model (in black):

![time_compare](https://user-images.githubusercontent.com/87817221/185794962-2a226796-2948-4b7e-9edf-e5419b8ebad6.png)

We could use this trade-off to integrate the desired model according to the purpose of our usage of the algorithm, but if we would have needed to choose one model which works good generally, we would integrate the model with the factor 0.241 for its good speed results and slight accuracy improvement.

This improvement shows that the use of such model in the FAsT-Match algorithm increases its stability in terms of speed, while keeping the same quality of results.

## How to Run This Project

In order to run the code, ones need to download the following packages:

* NumPy
* Matplotlib
* OpenCV
* scikit-learn
* pandas
* PyTorch
* shapely

This project can be run from PyCharm by running the main file and uncommenting the parts of the project which you want to activate in the `main` function in the function main.py.

```
# There are two options to test the algorithm, uncomment the one you would like to run.

images_folder = r"Images"  # Any image can be chosen
models_path = r"PyTorch_models"  # Choose which model you would like to test from the 'PyTorch_models' folder

# OPTION 1
# Run the FAsT-Match algorithm on an example image with a given/random template
'''
ex_image = cv2.imread(images_folder + "/thai_food.jpg")
ex_image = cv2.cvtColor(ex_image, cv2.COLOR_BGR2RGB)

# Given template:
ex_template = cv2.imread(...)
ex_real_corners = ...

# Random template:
ex_template, ex_real_corners = random_template(ex_image)

# With model
mlp_model = MLP.load_model(models_path + "/mlp0.241577.pth")
example_run(ex_image, ex_template, ex_real_corners, mlp_model)

# Without model
example_run(ex_image, ex_template, ex_real_corners)
'''

# OPTION 2
# Compare the original FAsT-Match algorithm to the algorithm with the improvement model
'''
mlp_model = MLP.load_model(models_path + "/mlp0.241577.pth")
img = cv2.imread(images_folder + "/zurich_object0024.view05.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
templates_amount = 3
check_model(templates_amount, img, mlp_model)
'''
```
