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
1. Upload Tiny VOC dataset to Google Colab.
2. Download the weights: https://drive.google.com/open?id=1CzR9Cu5A3zqfuhwTuq1Ia9bit-tHRqBC and upload them to Google Colab
3. Run Google Colab and choose in the settings 'GPU'.
4. Zip dataset and weights and upload them to Google Colab.
5. Run notebook file.

ИЗМЕНИТЬ

There are two section in notebook file: Testing and Training. For testing you need to upload one image to Google Colab and you can get speed evaluaion.
For training you need to use train dataset: https://github.com/penny4860/svhn-voc-annotation-format and testing data which you can get from train dataset (first 10000 images with annotations. It is optional).


Structure:
```
train
    - JPEGImages
    - Annotations
test
    - jpg
    - ann
```


P.S. If you have memory allocation problem, just decrease batch size. In my case it stopped working when batch size was more than 32
