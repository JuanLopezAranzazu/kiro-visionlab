# Requirements Document

## Introduction

VisionLab is an interactive computer vision playground built with Python, Streamlit, OpenCV, and NumPy. It allows users to upload an image and apply a sequential pipeline of visual transforms — including grayscale conversion, blur filters (Gaussian, Median, Bilateral), and edge detection algorithms (Canny, Sobel, Laplacian) — via an interactive sidebar with tabbed controls and per-transform parameter sliders. The application renders the original and processed images side by side and allows users to download the processed result.

## Glossary

- **App**: The VisionLab Streamlit web application.
- **User**: A person interacting with the App through a web browser.
- **Source Image**: The original image uploaded by the User, stored in memory for the duration of the session.
- **Processed Image**: The result of applying the active Transform Pipeline to the Source Image.
- **Transform**: A single image processing operation (e.g., grayscale conversion, Gaussian blur, Canny edge detection).
- **Transform Pipeline**: The ordered sequence of enabled Transforms applied left-to-right (top-to-bottom in the sidebar) to produce the Processed Image.
- **Sidebar**: The Streamlit sidebar panel containing tabbed controls for configuring Transforms.
- **Transform Tab**: A named tab within the Sidebar grouping related Transforms (e.g., "Blur", "Edges").
- **Parameter Slider**: A Streamlit slider widget that controls a numeric parameter for a specific Transform.
- **Side-by-Side View**: A two-column Streamlit layout showing the Source Image and the Processed Image simultaneously.
- **Download Button**: A Streamlit download_button widget that delivers the Processed Image as a PNG file.
- **Grayscale**: A Transform that converts the Source Image to a single-channel luminance representation.
- **Gaussian Blur**: A Transform that applies a Gaussian kernel to smooth the image; controlled by kernel size.
- **Median Blur**: A Transform that replaces each pixel with the median of its neighborhood; controlled by kernel size.
- **Bilateral Filter**: A Transform that smooths the image while preserving edges; controlled by diameter, sigma color, and sigma space.
- **Canny Edge Detection**: A Transform that detects edges using gradient thresholds; controlled by low threshold and high threshold.
- **Sobel Edge Detection**: A Transform that detects edges using horizontal and vertical Sobel kernels; controlled by kernel size.
- **Laplacian Edge Detection**: A Transform that detects edges using the Laplacian of the image; controlled by kernel size.
- **Session**: A single browser session with the App, lasting until the browser tab is closed or refreshed.

---

## Requirements

### Requirement 1 — Image Upload

**User Story:** As a User, I want to upload an image from my local machine, so that I can apply computer vision transforms to it.

#### Acceptance Criteria

1. THE App SHALL render a file uploader widget on the main page that accepts JPEG, JPG, and PNG file formats.
2. WHEN the User selects a valid image file, THE App SHALL decode the file into an in-memory NumPy array and store it as the Source Image for the current Session.
3. WHEN the User selects a valid image file, THE App SHALL display a success confirmation message indicating the image was loaded.
4. IF the User uploads a file with an unsupported format, THEN THE App SHALL display an error message stating the accepted formats and SHALL NOT update the Source Image.
5. WHILE no Source Image has been uploaded, THE App SHALL display a placeholder message instructing the User to upload an image.

---

### Requirement 2 — Side-by-Side Image View

**User Story:** As a User, I want to see the original and processed images next to each other, so that I can compare the effect of my transform settings at a glance.

#### Acceptance Criteria

1. WHEN a Source Image is present, THE App SHALL render the Side-by-Side View with two equal-width columns labeled "Original" and "Processed".
2. WHILE the Source Image is present, THE App SHALL display the Source Image in the "Original" column at full column width.
3. WHEN the Transform Pipeline produces a Processed Image, THE App SHALL display the Processed Image in the "Processed" column at full column width.
4. WHEN no Transforms are enabled, THE App SHALL display the Source Image in both columns.
5. WHEN the Processed Image is a single-channel (grayscale or edge-detected) image, THE App SHALL render it using a grayscale colormap so it displays correctly.

---

### Requirement 3 — Sidebar Tabbed Layout

**User Story:** As a User, I want the transform controls organized into tabs in the sidebar, so that I can navigate to the relevant category quickly without visual clutter.

#### Acceptance Criteria

1. THE App SHALL render the Sidebar with at least three Transform Tabs: "Color", "Blur", and "Edges".
2. THE App SHALL place the Grayscale Transform control in the "Color" tab.
3. THE App SHALL place all blur Transform controls (Gaussian Blur, Median Blur, Bilateral Filter) in the "Blur" tab.
4. THE App SHALL place all edge-detection Transform controls (Canny, Sobel, Laplacian) in the "Edges" tab.
5. WHILE the Sidebar is rendered, THE App SHALL show only the controls for the currently selected Transform Tab.

---

### Requirement 4 — Grayscale Transform

**User Story:** As a User, I want to convert the image to grayscale, so that I can simplify the color information before applying further transforms.

#### Acceptance Criteria

1. THE App SHALL render a toggle (checkbox or toggle widget) in the "Color" tab to enable or disable the Grayscale Transform.
2. WHEN the Grayscale Transform is enabled, THE App SHALL convert the Source Image (or prior pipeline output) to a single-channel luminance image using the standard OpenCV BGR-to-grayscale conversion.
3. WHEN the Grayscale Transform is disabled, THE App SHALL pass the image through unchanged.

---

### Requirement 5 — Blur Transforms

**User Story:** As a User, I want to apply blur filters with adjustable parameters, so that I can control the degree of smoothing applied to the image.

#### Acceptance Criteria

1. THE App SHALL render an enable toggle and a Parameter Slider for Gaussian Blur in the "Blur" tab; the slider SHALL control kernel size with odd integer values in the range [1, 31].
2. THE App SHALL render an enable toggle and a Parameter Slider for Median Blur in the "Blur" tab; the slider SHALL control kernel size with odd integer values in the range [1, 31].
3. THE App SHALL render an enable toggle and three Parameter Sliders for Bilateral Filter in the "Blur" tab: diameter in the range [1, 20], sigma color in the range [10, 150], and sigma space in the range [10, 150].
4. WHEN Gaussian Blur is enabled, THE App SHALL apply cv2.GaussianBlur to the current pipeline image using the selected kernel size.
5. WHEN Median Blur is enabled, THE App SHALL apply cv2.medianBlur to the current pipeline image using the selected kernel size.
6. WHEN Bilateral Filter is enabled, THE App SHALL apply cv2.bilateralFilter to the current pipeline image using the selected diameter, sigma color, and sigma space values.
7. WHEN a blur Transform is disabled, THE App SHALL pass the current pipeline image through unchanged.
8. IF a blur Transform is enabled and the current pipeline image is a single-channel image, THEN THE App SHALL apply the blur Transform to the single-channel image without channel conversion.

---

### Requirement 6 — Edge Detection Transforms

**User Story:** As a User, I want to apply edge detection algorithms with adjustable parameters, so that I can visualize structural features in the image.

#### Acceptance Criteria

1. THE App SHALL render an enable toggle and two Parameter Sliders for Canny Edge Detection in the "Edges" tab: low threshold in the range [0, 255] and high threshold in the range [0, 255].
2. THE App SHALL render an enable toggle and a Parameter Slider for Sobel Edge Detection in the "Edges" tab; the slider SHALL control kernel size with odd integer values in the range [1, 31].
3. THE App SHALL render an enable toggle and a Parameter Slider for Laplacian Edge Detection in the "Edges" tab; the slider SHALL control kernel size with odd integer values in the range [1, 31].
4. WHEN Canny Edge Detection is enabled, THE App SHALL apply cv2.Canny to the current pipeline image using the selected low and high threshold values.
5. WHEN Sobel Edge Detection is enabled, THE App SHALL compute the magnitude of the horizontal and vertical Sobel gradients using cv2.Sobel with the selected kernel size and combine them to produce the edge image.
6. WHEN Laplacian Edge Detection is enabled, THE App SHALL apply cv2.Laplacian to the current pipeline image using the selected kernel size and convert the result to an 8-bit unsigned image.
7. WHEN an edge detection Transform is disabled, THE App SHALL pass the current pipeline image through unchanged.
8. IF an edge detection Transform is enabled and the current pipeline image is a multi-channel (color) image, THEN THE App SHALL convert the image to grayscale before applying the edge detection algorithm.

---

### Requirement 7 — Sequential Transform Pipeline

**User Story:** As a User, I want transforms to stack sequentially, so that I can chain operations and see the cumulative effect.

#### Acceptance Criteria

1. THE App SHALL process the Transform Pipeline in a fixed order: Grayscale → Gaussian Blur → Median Blur → Bilateral Filter → Canny → Sobel → Laplacian.
2. WHEN the User changes any Parameter Slider or enable toggle, THE App SHALL recompute the full Transform Pipeline immediately and update the Processed Image.
3. WHEN only one Transform is enabled, THE App SHALL apply that single Transform to the Source Image and display the result.
4. WHEN no Transforms are enabled, THE App SHALL set the Processed Image equal to the Source Image.
5. THE App SHALL complete the full Transform Pipeline recomputation within 2 seconds of any User interaction that changes a transform parameter or toggle, given an image no larger than 4000×4000 pixels.

---

### Requirement 8 — Download Processed Image

**User Story:** As a User, I want to download the processed image, so that I can save and use the result outside the application.

#### Acceptance Criteria

1. WHEN a Source Image is present, THE App SHALL render a Download Button labeled "Download Processed Image" below the Side-by-Side View.
2. WHEN the User clicks the Download Button, THE App SHALL encode the Processed Image as a PNG file and deliver it to the browser as a file download named "processed_image.png".
3. WHEN no Transforms are enabled and the User clicks the Download Button, THE App SHALL deliver the Source Image encoded as a PNG file.
4. IF the Processed Image encoding fails, THEN THE App SHALL display an error message and SHALL NOT initiate a file download.

---

### Requirement 9 — Application Entry Point and Dependencies

**User Story:** As a developer, I want the application to have a clear entry point and declared dependencies, so that it can be installed and run in a consistent environment.

#### Acceptance Criteria

1. THE App SHALL provide a single Python entry point file named `app.py` at the repository root that can be launched with `streamlit run app.py`.
2. THE App SHALL declare all runtime dependencies in a `requirements.txt` file at the repository root, including pinned or minimum-version entries for streamlit, opencv-python, and numpy.
3. WHILE the App is running, THE App SHALL not require any configuration beyond installing the dependencies listed in `requirements.txt`.
