# Why is there no distinction between RGB and BGR in IEM?

The difference between RGB and GBR format is due to OpenCV. In IEM, we do not distinguish between the two because they have high similarity and similar performance. We use the IEM teacher to extract features of RGB and BGR images and calculate the similarity between feature pairs, as shown in the following figure:

![image](./similarity.png)

We found the features from RGB and BGR images have high similarity. In addition, we also evaluated the performance of RGB features and BGR features extracted by IEM teacher under linear probing:

|      | Average ROC-AUC on 8 classification tasks | Average RMSE on 4 regression tasks |
| ---- | ----------------------------------------- | ---------------------------------- |
| BGR  | **64.66**                                 | 1.406                              |
| RGB  | 64.33                                     | **1.369**                          |

We found that IEM is not sensitive to RGB and BGR, so we did not distinguish between RGB and BGR images. However, we still recommend that you convert the image to RGB format for uniformity.

