import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.neural_network import MLPClassifier


# -------------------------------
# Load Image
# -------------------------------

img_file = "Stone_1.jpg"

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()

original = img.copy()



# -------------------------------
# Convert to Gray-Image
# -------------------------------

# Create an iterator from the uploaded files dictionary
itr = iter(img_file)
# Get the first filename from the iterator
# This is the name of the uploaded image file
filename = next(itr)
# Read the image from the file using OpenCV
# By default, OpenCV reads images in BGR format
ORG_IMG = cv2.imread(filename)
# Convert the image from BGR color space to RGB color space
ORG_IMG = cv2.cvtColor(ORG_IMG, cv2.COLOR_BGR2RGB)
# Convert the RGB image to Grayscale
gray_img = cv2.cvtColor(ORG_IMG, cv2.COLOR_RGB2GRAY)

# Display Image
plt.figure(figsize=(6, 6))
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.title('Gray Image')
plt.show()


# -------------------------------
# Apply Median Filter
# -------------------------------

# Apply Median Filtering to the grayscale image
# Median filter is used to remove speckle noise and salt-and-pepper noise
# The kernel size is 5x5 (must be an odd number)
filtered_image = cv2.medianBlur(gray_img, 5)

# Display Image
plt.figure(figsize=(6, 6))
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')
plt.title('Filter Image')
plt.show()


# -------------------------------
# Apply DWT Preprocessing
# -------------------------------

def apply_dwt_preprocessing(filtered_image):

    # Apply Discrete Wavelet Transform (DWT)
    # dividing image into 4 sub-bands: LL, LH, HH, HL.
    # 'haar' is a common wavelet.
    coeffs = pywt.dwt2(filtered_image, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Process the LL Band (Approximation)
    # The LL band contains the low-frequency structural information (denoised).
    # DWT reduces the image size by half.

    # Normalize LL band to 0-255 range for image representation
    LL_norm = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
    LL_img = np.uint8(LL_norm)

    # Resize back to original image size
    preprocessed_image = cv2.resize(LL_img, (filtered_image.shape[1], filtered_image.shape[0]))
    return preprocessed_image

# Function Call
preprocessed_image = apply_dwt_preprocessing(filtered_image)

# Display Image
plt.figure(figsize=(6, 6))
plt.imshow(preprocessed_image, cmap='gray')
plt.axis('off')
plt.title('preprocessed Image')
plt.show()



# -------------------------------
# Apply Fuzzy C-Mean Clustering
# -------------------------------

def apply_fuzzy_cmean(preprocessed_image):
  # 2D image converted to 1D array
  flatended = np.array(preprocessed_image).flatten()
  # Normalizing the image for mathmatical calculations
  normalized = flatended / flatended.max()
  clusters=5 # Using 5 clusters (5 levels of intensities)
  fuzziness_coefficient=2 # Fuzziness Coefficient controls how fuzzy the clusters will be

  # Declare Random probability for Membership matrix for each cluster so that sum of all cluster probability for each pixel is 1
  membership_matrix = np.random.random(size=(len(normalized), clusters)) # 2D array with rows= flatended Image size, cols= number of clusters
  row_sums= membership_matrix.sum(axis=1, keepdims=True)
  membership_matrix= membership_matrix / row_sums # each row element divided by row sum, so that new Row sum= 1

  # There will be 1 centroid for each cluster
  cluster_centroid = np.zeros(clusters)
  max_iterations = 500 # Limiting the iterations to a number to protecting function from infinite loop and time constrains

  for i in range(max_iterations):
    # calculate cluster centroids (average intensities of each clusters)
    weight = membership_matrix ** fuzziness_coefficient # Get the weighted value
    numerator = np.dot(weight.T,   normalized) # sum(weight * pixel_value) --> 1D array of length clusters
    denominator = weight.sum(axis=0) #  1D array with sum of all weights for each cluster
    cluster_centroid = numerator / denominator # cluster centroids

    # Calculate distance matrix
    distance_matrix = np.zeros((len(normalized), clusters)) # Distance Matrix --> Same shape as membership metrix
    for c in range(clusters):
        distance_matrix[:, c] = np.abs(normalized - cluster_centroid[c]) # Distance of pixel intensity from the centroids
        distance_matrix = np.fmax(distance_matrix, 1e-10) # If the distance is 0, We replace it with 1e-10 to avoid devide by 0
    power = 2/(fuzziness_coefficient - 1) #2/(m-1)
    inv_distance_power= 1/(distance_matrix ** power) # 1/(d_ij)^(2/(m-1))
    row_sums = inv_distance_power.sum(axis=1, keepdims=True) #sum over clusters for each data point
    U_new = inv_distance_power / row_sums # Updated membership matrix

    diff = np.linalg.norm(U_new - membership_matrix)
    if(diff < 1e-5):
      break
    else:
      membership_matrix = U_new

  print("Converged in {} iterations".format(i+1))

  # Take the index of maximum probability for each pixel and reshape it into the image shape
  labels = np.argmax(membership_matrix, axis=1) # identifies each pixel is from which cluster through cluster index (maximum membership is the cluster the pixel is from)
  segmented_image = labels.reshape(preprocessed_image.shape)

  # Set the stone cluster  as the maximum of the all cluster centroids and make a binary mask of the stone cluster
  stone_cluster_index = np.argmax(cluster_centroid) # The index of the maximum intensity is the stone cluster
  binary_mask = (segmented_image == stone_cluster_index).astype(np.uint8) * 255 # if the segmented image has stone cluster index, then 255, else 0
  return binary_mask

# Function Call
masked_stone = apply_fuzzy_cmean(preprocessed_image)

# Display Image
plt.figure(figsize=(6, 6))
plt.imshow(masked_stone, cmap='gray')
plt.axis('off')
plt.title('Masked Image')
plt.show()


# -------------------------------
# Apply Morphology to refine edges and filter out noices
# -------------------------------

def apply_morphology(masked_stone):
  # Taking 5X5 kernel for morphological analysis
  kernel = np.ones((5, 5), np.uint8)
  # Apply both opening and closing
  clean_mask = cv2.morphologyEx(masked_stone, cv2.MORPH_OPEN, kernel) #Opening --> to reduce specles and noices
  clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel) # Closing --> to reduce the gaps inside a mask
  contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Get the contours(bounderies) of the masks
  img_copy = ORG_IMG.copy()
  for contour in contours:
    cv2.drawContours(img_copy, contour, -1, (0,255,0), 2) # draw on the copied image to specify mask location drawContours(image, contour, -1(to cover full shape), color, thickness)
  return contours,clean_mask,img_copy


stone_contours, clean_mask, drawn_contours = apply_morphology(masked_stone)

# Display Image
# 1. Refined Mask Image
plt.figure(figsize=(20,20))
plt.subplot(1, 3, 1)
plt.title("Refined Mask")
plt.imshow(clean_mask, cmap='gray')
plt.axis('off')

# 2. Selected Areas Image
plt.subplot(1, 3, 2)
plt.title("Selected Areas")
plt.imshow(drawn_contours)
plt.axis('off')


# -------------------------------
# BPNN Training: Clone Github Repository to load the pre-calculated features and labels for saving time
# -------------------------------

# ---------------- Step 1: Download the Preprocessed Dataset ----------------

# # Clone the dataset repository from GitHub
# if os.path.exists('/content/kidney-stone-images'):
#     !rm -rf /content/kidney-stone-images
# !git clone https://github.com/Thejellyfish1024/kidney-stone-images.git


# ---------------- Step 2: Load Pre-Extracted Features ----------------
# x_train = np.load('/content/kidney-stone-images/x_train_bin.npy')
# y_train = np.load('/content/kidney-stone-images/y_train_bin.npy')


# -------------------------------
# BPNN Training: Clone Github Repository to load the pre-calculated features and labels for saving time
# -------------------------------

# This function applies the same preprocessing pipeline used in testing, so that training and testing remain consistent
def full_preprocess_for_training(gray):

    # Apply Median Filter to remove speckle and impulse noise
    blur = cv2.medianBlur(gray, 5)

    # Apply Discrete Wavelet Transform (DWT) and keep LL band
    # This helps in noise reduction and feature enhancement
    dwt_img = apply_dwt_preprocessing(blur)

    # Apply Fuzzy C-Means (FCM) clustering for segmentation
    # This separates possible stone regions from background
    mask = apply_fuzzy_cmean(dwt_img)

    # Morphological Processing
    # Create a 5x5 structuring element (kernel) for morphology
    kernel = np.ones((5,5), np.uint8)
    # Apply Opening operation to remove small noise particles
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Apply Closing operation to fill small holes inside regions
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    #Contour Detection
    # Find external contours from the cleaned binary mask
    # These contours represent candidate stone regions
    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Return the detected contours for feature extraction
    return contours

# -------------------------------
# Optional(Part-2): BPNN Training -> Feature Extraction and Back Propagation Neural Network (BPNN) Training for Classification.
# -------------------------------

# This function extracts texture features from training images and prepares data for neural network classification.
def train_features(folder_path, label):
    # List to store extracted feature vectors
    features_list = []

    # List to store corresponding class labels
    labels_list = []

    # Get all image filenames from the given folder
    images = [f for f in os.listdir(folder_path)
              if f.endswith(('.jpg','.png','.jpeg'))]

    # Process each training image one by one
    for img_name in images:

        # Create full file path for the image
        full_path = os.path.join(folder_path, img_name)

        # Read image in grayscale mode
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        # Skip file if image loading fails
        if img is None:
            continue

        # Apply the complete preprocessing pipeline
        # (Median → DWT → FCM → Morphology → Contours)
        contours = full_preprocess_for_training(img)

        # Process each detected contour region
        for cnt in contours:

            # Ignore very small regions (likely noise)
            if cv2.contourArea(cnt) > 100:

                # Get bounding box of the contour
                x, y, w, h = cv2.boundingRect(cnt)

                # Extract Region of Interest (ROI)
                roi = img[y:y+h, x:x+w]

                # Skip empty regions
                if roi.size == 0:
                    continue

                # ---------------- GLCM Feature Extraction ----------------

                # Compute Gray Level Co-occurrence Matrix (GLCM)
                glcm = graycomatrix(
                    roi,
                    distances=[1],        # Distance between pixels
                    angles=[0],           # Horizontal direction
                    levels=256,           # Gray levels
                    symmetric=True,       # Make matrix symmetric
                    normed=True           # Normalize matrix
                )

                # Extract texture features from GLCM
                contrast = graycoprops(glcm,'contrast')[0,0]
                energy = graycoprops(glcm,'energy')[0,0]
                homogeneity = graycoprops(glcm,'homogeneity')[0,0]

                # Store extracted features
                features_list.append([
                    contrast,
                    energy,
                    homogeneity
                ])

                # Store corresponding class label
                # 1 = Stone, 0 = Normal
                labels_list.append(label)

    # Return all features and labels for training
    return features_list, labels_list

# Path to stone and normal image folders
stone_path = '/content/kidney-stone-images/Stone'
normal_path = '/content/kidney-stone-images/Normal'

# Extract features from stone images (label = 1)
stone_features, stone_labels = train_features(stone_path, 1)

# Extract features from normal images (label = 0)
normal_features, normal_labels = train_features(normal_path, 0)

# Combine both classes into a single dataset
x_train = np.array(stone_features + normal_features)
y_train = np.array(stone_labels + normal_labels)

# Initialize Backpropagation Neural Network (MLP Classifier)
# Two hidden layers with 10 neurons each
# Maximum 2000 training iterations
bpnn_model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    max_iter=2000
)

# Train the neural network using extracted features
bpnn_model.fit(x_train, y_train)

# -------------------------------
# Extracting Features for BPNN Classification
# -------------------------------

def extract_features(contour, image):
    # Step 1: Create an empty binary mask with same size as image
    mask = np.zeros(image.shape, dtype=np.uint8)
    # Draw the given contour on the mask
    # -1 means draw all points of the contour
    # 255 means white (selected region)
    # -1 thickness means fill the contour completely
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Step 2: Find Bounding Box of the Contour
    # This helps in cropping only the region of interest (ROI)
    x, y, w, h = cv2.boundingRect(contour)

    # Step 3: Extract Region of Interest (ROI)
    # Crop the image using bounding box coordinates
    # Boundary checks are used to avoid index errors
    roi = image[
        max(0, y) : min(image.shape[0], y + h),
        max(0, x) : min(image.shape[1], x + w)
    ]

    # If ROI is empty, return zero features (safety check)
    if roi.size == 0:
        return [0, 0, 0]

    # Step 4: Compute Gray Level Co-occurrence Matrix (GLCM)
    # GLCM represents texture information of the ROI
    glcm = graycomatrix(
        roi,                # Input grayscale region
        distances=[1],      # Pixel distance (1 = nearest neighbor)
        angles=[0],         # Horizontal direction (0 degrees)
        levels=256,         # Number of gray levels (0–255)
        symmetric=True,     # Make matrix symmetric
        normed=True         # Normalize values
    )

    # Step 5: Extract Texture Features from GLCM
    # These features describe surface characteristics

    # Contrast: Measures local intensity variation
    contrast = graycoprops(glcm, 'contrast')[0, 0]

    # Energy: Measures uniformity of texture
    energy = graycoprops(glcm, 'energy')[0, 0]

    # Homogeneity: Measures closeness of pixel distribution
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Step 6: Return Feature Vector
    return [contrast, energy, homogeneity]

# -------------------------------
# Final Stone Detection with and Classification Using BPNN
# -------------------------------

def draw_marks():
    # Create a copy of the original RGB image for final visualization
    result_image = ORG_IMG.copy().astype(np.uint8)

    # Check if the image is grayscale (2D array)
    # If yes, convert it to BGR/RGB format for drawing colored boxes
    if len(result_image.shape) == 2:

        result_image = cv2.cvtColor(
            result_image,
            cv2.COLOR_GRAY2BGR
        ).astype(np.uint8)

    # Boolean flag to check whether any stone is detected
    found_any = False

    # Loop through all contours obtained after morphological processing
    # These contours represent possible stone regions
    for contour in stone_contours:
        # Calculate area (number of pixels) of the contour
        area = cv2.contourArea(contour)

        # Calculate perimeter (boundary length) of the contour
        perimeter = cv2.arcLength(contour, True)

        # Ignore very small regions and invalid shapes
        # This helps remove noise and false detections
        if perimeter == 0 or area < 100:
            continue

        # Step 1: Shape Analysis using Circularity
        # --------------------------------------------------------
        # Calculate circularity using the standard formula
        # Circularity = (4 * π * Area) / (Perimeter²)
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Step 2: Physical Filtering based on Size and Shape
        # --------------------------------------------------------
        # Only consider regions that:
        # - Have reasonable size (150–2000 pixels)
        # - Have circular shape (circularity > 0.5)
        # This removes tissues and background structures
        if 150 < area < 2000 and circularity > 0.5:

            # Step 3: Texture-Based Classification using BPNN
            # --------------------------------------------------------
            # Extract GLCM texture features from the region
            features = extract_features(contour, gray_img)

            # Predict the class using trained neural network
            # Output: 1 = Stone, 0 = Normal
            prediction = bpnn_model.predict([features])

            # If the classifier confirms this region as a stone
            if prediction[0] == 1:
                # Mark that at least one stone is detected
                found_any = True

                # Get bounding rectangle of the detected stone
                x, y, w, h = cv2.boundingRect(contour)

                # Draw green bounding box around detected stone
                cv2.rectangle(
                    result_image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Add text label "Stone" above the bounding box
                cv2.putText(
                    result_image,
                    "Stone",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # Return the final image with detected stone regions
    return result_image.astype(np.uint8)

# Call the function to generate final detected output image
final_image = draw_marks()


# -------------------------------
# Display Final Output
# -------------------------------

plt.figure(figsize=(10, 10))

# 1. Original Image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(ORG_IMG)
plt.axis('off')

# 2. Final Detection
plt.subplot(2, 2, 2)
plt.title("Final Detection")
plt.imshow(final_image)
plt.axis('off')


plt.tight_layout()
plt.show()