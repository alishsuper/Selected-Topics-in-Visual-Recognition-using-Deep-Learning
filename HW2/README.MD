## HW2: Celebrity face images generation

### [Slides](https://docs.google.com/presentation/d/1_J_R4PSI0KtyGWucHfL3uIfsU1D-xdb2BmHhhIspHDE/edit)

### Goals
Train your generative adversiral network to generat the celebrity face images as real as possible

### Description
- Download the dataset from code provide in [homework2.ipynb](https://github.com/NCTU-VRDL/CS_IOC5008/blob/master/HW2/homework2.ipynb)
- After trained your GAN, generated **500 images** (each image contains 3x3 grid of images) and save it by the "output_fig" function below, **Please note that the width, height of your image should be within the range [28, 112]**
```python
def output_fig(images_array, file_name="./results"):
    # the shape of your images_array should be (9, width, height, 3),  28 <= width, height <= 112 
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)
```
- This homework will be evaluated by your classmates through voting for realistic (compare to baseline) in class
- Rank top3  will be invited to make a presentation to share your methodology and get a bonus on final score!

### Implementing
1. Download celebA dataset: https://www.kaggle.com/jessicali9530/celeba-dataset and change the directory in jupyter file.
2. Download the weights: https://drive.google.com/drive/folders/1ifnUYLriCpdq0DnTOH4SV1KLCywVzPqr and put the folder "model" to HW2.
3. Run jupyter file.
4. helper.py file provides the functions which we will use in the main program.
5. problem_unittests.py provides unittests.
