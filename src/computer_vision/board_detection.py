import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread(r"assets/board.png")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Histogram equalization for contrast enhancement
hist_eq = cv2.equalizeHist(gray_image)
# Large Gaussian blur for noise reduction and edge enhancement
blur = cv2.GaussianBlur(hist_eq, (21, 21), 0)

plt.figure()
plt.title("Histogram Equalized + Blurred")
plt.imshow(blur, cmap="gray")
plt.axis("off")
plt.show()

print(f"Image shape: {blur.shape}")  # Diagnostic: check image size


def detect_and_draw_circles(input_img, display_title):
    circles = cv2.HoughCircles(
        input_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=80,
        param2=20,  # Lower for more sensitivity on blurred image
        minRadius=40,  # Set based on your slot size in pixels
        maxRadius=60,
    )
    board_circles = rgb_image.copy()
    circle_centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        print(f"Detected {len(circles)} circles in {display_title}")
        for i in circles:
            center = (i[0], i[1])
            radius = i[2]
            circle_centers.append((i[0], i[1], radius))
            cv2.circle(board_circles, center, radius, (0, 255, 0), 2)
            cv2.circle(board_circles, center, 2, (0, 0, 255), 3)
    else:
        print(f"No circles detected in {display_title}.")
    plt.figure()
    plt.title(display_title)
    plt.imshow(board_circles)
    plt.axis("off")
    plt.show()
    return circle_centers


# Use the histogram-equalized, blurred image for circle detection
circle_centers = detect_and_draw_circles(
    blur, "Board Circles Detection (HistEq + Blurred Input)"
)


def sort_grid_centers(centers, grid_size=4):
    # Sort by y, then by x within each row
    centers = sorted(centers, key=lambda c: (c[1], c[0]))
    rows = [centers[i * grid_size : (i + 1) * grid_size] for i in range(grid_size)]
    for row in rows:
        row.sort(key=lambda c: c[0])
    # Flatten back to list
    sorted_centers = [c for row in rows for c in row]
    return sorted_centers


# Detect circles
circle_centers = detect_and_draw_circles(
    gaussian_blur, "Board Circles Detection (Gaussian Blur Input)"
)

if len(circle_centers) == 16:
    # Sort centers into 4x4 grid order
    sorted_centers = sort_grid_centers(circle_centers, grid_size=4)
    labeled_img = rgb_image.copy()
    for idx, (x, y, r) in enumerate(sorted_centers):
        row = idx // 4
        col = idx % 4
        # Draw semi-transparent mask
        overlay = labeled_img.copy()
        cv2.circle(overlay, (x, y), r, (0, 255, 255), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, labeled_img, 1 - alpha, 0, labeled_img)
        # Draw label
        cv2.putText(
            labeled_img,
            f"({row},{col})",
            (x - 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
    plt.figure()
    plt.title("Labeled Board Spots (row, col)")
    plt.imshow(labeled_img)
    plt.axis("off")
    plt.show()
else:
    print(
        f"Expected 16 circles, but detected {len(circle_centers)}. Adjust detection parameters if needed."
    )
