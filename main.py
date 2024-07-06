import cv2
import numpy as np
import dlib

# Load dlib's face detector and facial landmark predictor
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(image, filename):
  """ Detect facial landmarks in the image and save an image with landmarks overlay. """
  rects = DETECTOR(image, 1)
  if len(rects) > 1:
      raise Exception('Too many faces detected.')
  elif len(rects) == 0:
      raise Exception('No faces detected.')
  landmarks = np.array([[p.x, p.y] for p in PREDICTOR(image, rects[0]).parts()])
  image_with_landmarks = image.copy()
  for (x, y) in landmarks:
      cv2.circle(image_with_landmarks, (x, y), 2, (0, 255, 0), -1)  # Draw green dots for landmarks
  cv2.imwrite(f'{filename}_landmarks.jpg', image_with_landmarks)
  return landmarks

def transformation_from_points(points1, points2):
  """ Compute affine transformation matrix from points1 to points2. """
  points1 = points1.astype(np.float64)
  points2 = points2.astype(np.float64)
  c1, c2 = np.mean(points1, axis=0), np.mean(points2, axis=0)
  points1 -= c1
  points2 -= c2
  s1, s2 = np.std(points1), np.std(points2)
  points1 /= s1
  points2 /= s2
  U, _, Vt = np.linalg.svd(points1.T @ points2)
  R = (U @ Vt).T
  return np.hstack([(s2 / s1) * R, (c2.T - (s2 / s1) * R @ c1.T)[:, None]])

def create_mask(points, shape):
   """ Create a binary mask from the points. """
   mask = np.zeros(shape[:2], dtype=np.float32)
   convexhull = cv2.convexHull(points)
   cv2.fillConvexPoly(mask, convexhull, 1)
   mask = cv2.GaussianBlur(mask, (11, 11), 5)
   # Convert mask to 3 channels
   mask_3c = np.stack([mask] * 3, axis=-1)
   return mask_3c

def warp_face(image, M, dshape):
  """ Warp the image using the affine transformation. """
  return cv2.warpAffine(image, M[:2], (dshape[1], dshape[0]), borderMode=cv2.BORDER_TRANSPARENT)

def color_correction(face1, face2, mask):
   """ Perform color correction on face1 to match the color tones of face2. """
   # Convert faces to LAB color space
   face1_lab = cv2.cvtColor(face1, cv2.COLOR_BGR2LAB)
   face2_lab = cv2.cvtColor(face2, cv2.COLOR_BGR2LAB)

   # Extract L, A, B channels from both faces
   l1, a1, b1 = cv2.split(face1_lab)
   l2, a2, b2 = cv2.split(face2_lab)


   # Perform color correction on each channel
   l_corrected = cv2.addWeighted(l1, 0.8, l2, 0.2, 0)
   a_corrected = cv2.addWeighted(a1, 0.8, a2, 0.2, 0)
   b_corrected = cv2.addWeighted(b1, 0.8, b2, 0.2, 0)


   # Merge corrected LAB channels
   corrected_lab = cv2.merge([l_corrected, a_corrected, b_corrected])


   # Convert back to BGR color space
   corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)


   # Blend corrected face with face2 using the mask
   corrected_combined = (corrected_bgr * mask + face2 * (1 - mask)).astype(np.uint8)


   return corrected_combined


def process_images(face1_path, face2_path):
  """ Load images, process them, and save the output. """
  face1_im = cv2.imread(face1_path)
  face2_im = cv2.imread(face2_path)
  face1_points = get_landmarks(face1_im, 'face1')
  face2_points = get_landmarks(face2_im, 'face2')


  M = transformation_from_points(face1_points, face2_points)
  warped_face1 = warp_face(face1_im, M, face2_im.shape)
  mask = create_mask(face2_points, face2_im.shape)
  combined_im = color_correction(warped_face1, face2_im, mask)


  cv2.imwrite('combined.jpg', combined_im)


# Run the face swap process
process_images('face1.jpg', 'face2.jpg')
