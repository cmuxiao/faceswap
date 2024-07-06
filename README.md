# Face Swap with Landmark Detection and Color Correction

This project provides a simplified version of face swap I did for work. The script performs face swap between two images using facial landmark detection, affine transformation, and color correction. It uses OpenCV and dlib libraries for image processing and facial landmark detection.

## Prerequisites

Before you begin, ensure you have the following libraries installed:

- OpenCV
- dlib
- numpy

You will also need the `shape_predictor_68_face_landmarks.dat` file, which contains the facial landmark predictor model.

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/cmuxiao/faceswap.git
cd faceswap
```

2. **Install required libraries:**

```bash
pip install opencv-python dlib numpy
```

3. **Download the facial landmark predictor:**

Download the `shape_predictor_68_face_landmarks.dat` file from [this link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place it in the project directory.

## Usage

1. **Prepare the input images:**

Place the two input images (`face1.jpg` and `face2.jpg`) in the project directory.

2. **Run the face swap process:**

```bash
python faceswap.py
```

This will process the images and save the output as `combined.jpg`.

## Script Breakdown

### get_landmarks(image, filename)

Detects facial landmarks in the input image and saves an image with the landmarks overlay.

### transformation_from_points(points1, points2)

Computes the affine transformation matrix from `points1` to `points2`.

### create_mask(points, shape)

Creates a binary mask from the given points.

### warp_face(image, M, dshape)

Warps the image using the affine transformation matrix `M`.

### color_correction(face1, face2, mask)

Performs color correction on `face1` to match the color tones of `face2`.

### process_images(face1_path, face2_path)

Loads the input images, processes them, and saves the output image.

## Example

Input images:

- `face1.jpg`: ![face1](images/face1.jpg)
- `face2.jpg`: ![face2](images/face2.jpg)

Output image:

- `combined.jpg`: ![combined](images/combined.jpg)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [dlib library](http://dlib.net/)
- [OpenCV library](https://opencv.org/)
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

Feel free to modify and expand upon this script to suit your specific needs.
