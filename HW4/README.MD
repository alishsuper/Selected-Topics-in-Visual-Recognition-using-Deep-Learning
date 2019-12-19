## HW4: Tiny Pascal Dataset Instance segmentation

### Slides
https://docs.google.com/presentation/d/1U1gx48DTkEAJAOE7ycSCcZVlfBdQ0LLEb5X6QF0jKMs/edit#slide=id.g7a5525b70b_0_115
### Goals
Train your convolution neural network to segment the common objects in image

### Google Drive url
https://drive.google.com/drive/u/3/folders/1kwDlhY3Fz346fA2csCDG0vSDMTP10WvX

### Description
- Upload your submission json file in the submission folder, I will inference it and update the results on your filename for **every midnight (23:59)!**
- Rank top3 on the leaderboard will be invited to make a presentation to share your methodology and get a bonus on your final score!

### Implementation
Notebook file was created in Google Colab
1. Download Tiny Pascal VOC train dataset: https://drive.google.com/open?id=1MQD3pDDBk6X7Xz8Y8OKnDIU-7pyVUhAI and upload it to Google Colab
2. Download the weights: https://drive.google.com/open?id=1CzR9Cu5A3zqfuhwTuq1Ia9bit-tHRqBC and upload them to Google Colab
3. Run Google Colab and choose in the settings 'GPU'.
4. Zip dataset and weights and upload them to Google Colab.
5. Run notebook file.
6. Follow to the comments in notebook file. It is very IMPORTANT!

Structure:
```
train_images
    - JPEGImages
test_images
    - JPEGImages
pascal_train.json
test.json
```

P.S. Try to use this: cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml").
It will give higher scores
