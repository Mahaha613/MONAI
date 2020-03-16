# Modules for public alpha

MONAI aims at supporting deep learning in medical image analysis at multiple granuality.
This figure shows modules currently available in the codebase.
![image](../images/end_to_end_process.png)
The rest of this page provides more details for each module.

* [Image transformations](#image-transformations)
* [Loss functions](#losses)
* [Network architectures](#network-architectures)
* [Evaluation](#evaluation)
* [Visualization](#visualization)
* [Result writing](#result-writing)

## Image transformations
Medical image data pre-processing is challenging.  Data are often in specialized formats with rich meta information;  and the data volumes are often high-dimensional and requiring carefully designed manipulation procedures. As an important part of MONAI, powerful and flexible image transformations are provided to enable user-friednly, reproducible, optimized medical data pre-processing piepline.

### 1. Transforms support both Dictionary and Array format data
1. The widely used computer vision packages (such as ``torchvision``) focus on spatially 2D array image processing. MONAI provides more domain specific transformations for both sptially 2D and 3D, and retains the flexible transformation "compose" feature.
2.  As medical image preprocessing often requires additional fine-grained system parameters, MONAI provides transforms for input data encapsulated in a python dictionary. Users are able to specify the keys corresponding to the expected data fields and system parameters to compose complex transformations.

### 2. Medical specific transforms
MONAI aims at providing a rich set of popular medical image specific transformamtions. These currently include, for example:


- `LoadNifti`:  Load Nifti format file from provided path
- `Spacing`:  Resample input image into the specified `pixdim`
- `Orientation`: Change image's orientation into the specified `axcodes`
- `GaussianNoise`: Pertubate image intensities by adding statistical noises
- `IntensityNormalizer`: Intensity Normalization based on mean and standard deviation
- `Affine`: Transform image based on the affine parameters
- `Rand2DElastic`: Random elastic deformation and affine in 2D
- `Rand3DElastic`: Random elastic deformation and affine in 3D

### 3. Fused spatial transforms and GPU acceleration
As medical image volumes are usually large (in multi-dimensional arrays), pre-processing performance obviously affects the overall pipeline speed. MONAI provides affine transforms to execute fused spatial operations, supports GPU acceleration via native PyTorch to achieve high performance.
Example code:
```py
# create an Affine transform
affine = Affine(
    rotate_params=np.pi/4,
    scale_params=(1.2, 1.2),
    translate_params=(200, 40),
    padding_mode='zeros',
    device=torch.device('cuda:0')
)
# convert the image using interpolation mode
new_img = affine(image, spatial_size=(300, 400), mode='bilinear')
```
### 4. Randomly crop out batch images based on positive/negative ratio
Medical image data volume may be too large to fit into GPU memory. A widely-used approach is to randomly draw small size data samples during training. MONAI currrently provides uniform random sampling strategy as well as class-balanced fixed ratio sampling which may help stabilize the patch-based training process.

## Losses
There are domain-specific loss functions in the medical research area which are different from the generic computer vision ones. As an important module of MONAI, these loss functions are implemented in PyTorch, such as Dice loss and generalized Dice loss.

## Network architectures
Some deep neural network architectures have shown to be particularly effective for medical imaging analysis tasks. MONAI implements reference networks with the aims of both flexibility and code readability.

## Evaluation
To run model inferences and evaluate the model quality, MONAI provides reference implementation for the relevant widely-used approaches. Currently several popular evaluation metrics and inference patterns are included:

### 1. Sliding window inference
When executing inference on large medical images, the sliding window is a popular method to achieve high performance with flexible memory requirements.
1. Select continuous windows on the original image.
2. Execute a batch of windows on the model per time, and complete all windows.
3. Connect all the model outputs to construct one segmentation corresponding to the original image.
4. Save segmentation result to file or compute metrics.
![image](../images/sliding_window.png)

### 2. Metrics for medical tasks
There are many useful metrics to measure medical specific tasks, MONAI already implemented Mean Dice and AUC, will integrate more soon.

## Visualization
Besides common curves of statistics on TensorBoard, in order to provide straight-forward checking of 3D image and the corresponding label and segmentation output, MONAI can visualize 3D data as GIF animation on TensorBoard which can help users quickly check model output.

## Result writing
For the segmentation task, MONAI supports to save model output as NIFTI format image and add affine information from the corresponding input image.

For the classification task, MONAI supports to save classification result as a CSV file.
