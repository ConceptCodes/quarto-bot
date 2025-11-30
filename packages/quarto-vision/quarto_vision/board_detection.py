import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_board_boundary(image, gray):
    """Detect the outer circular boundary of the Quarto board."""

    # Apply Gaussian blur for better circle detection
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detect the outer board circle (larger radius range)
    board_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        gray.shape[0] // 4,  # Only expect one large circle
        param1=50,
        param2=30,
        minRadius=int(min(gray.shape) * 0.2),  # At least 20% of image
        maxRadius=int(min(gray.shape) * 0.6),  # At most 60% of image
    )

    if board_circles is not None:
        # Take the largest circle (most likely the board)
        board_circles = np.round(board_circles[0, :]).astype("int")
        largest_circle = max(board_circles, key=lambda x: x[2])  # Sort by radius
        return largest_circle

    return None


def detect_game_circles(image, gray, board_circle):
    """Detect the 16 game position circles within the board boundary."""

    if board_circle is None:
        print("No board detected, using full image")
        return None

    board_x, board_y, board_radius = board_circle

    # Calculate expected game circle parameters based on board size
    # Your measurements: 236mm board, 28mm circles, 4mm gaps
    pixels_per_mm = (board_radius * 2) / 236  # 236mm board diameter

    game_circle_radius = int((28 / 2) * pixels_per_mm)  # 14mm radius
    min_distance = int((28 + 4) * pixels_per_mm)  # 32mm center-to-center

    # Create a mask for the board area (slightly smaller to avoid edges)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (board_x, board_y), int(board_radius * 0.85), 255, -1)

    # Apply mask to the grayscale image
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Detect circles within the board area
    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        1,
        min_distance,
        param1=50,
        param2=25,  # Lower threshold to detect more circles
        minRadius=max(1, int(game_circle_radius * 0.7)),
        maxRadius=int(game_circle_radius * 1.3),
    )

    print(f"Board radius: {board_radius} pixels")
    print(f"Pixels per mm: {pixels_per_mm:.2f}")
    print(f"Expected game circle radius: {game_circle_radius} pixels")
    print(f"Min distance between circles: {min_distance} pixels")

    return circles


# Load and process image
image = cv2.imread(
    r"data/collected_images/img_0092_1752193320241.jpg", cv2.IMREAD_COLOR
)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce noise
gray = cv2.medianBlur(gray, 5)

# Step 1: Detect board boundary
print("Detecting board boundary...")
board_circle = detect_board_boundary(image, gray)

if board_circle is not None:
    board_x, board_y, board_radius = board_circle
    print(f"Board detected at ({board_x}, {board_y}) with radius {board_radius}")

    # Draw board boundary in blue
    cv2.circle(image, (board_x, board_y), board_radius, (255, 0, 0), 3)
    cv2.circle(image, (board_x, board_y), 5, (255, 0, 0), -1)
else:
    print("Board boundary not detected!")

# Step 2: Detect game circles within board
print("Detecting game circles...")
circles = detect_game_circles(image, gray, board_circle)

# Draw detected game circles
CIRCLE_CENTER_COLOR = (0, 100, 100)  # Cyan
CIRCLE_OUTLINE_COLOR = (255, 0, 255)  # Magenta

if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Detected {len(circles[0])} game circles")

    for i, circle in enumerate(circles[0, :]):
        center = (circle[0], circle[1])
        radius = circle[2]

        # Draw circle center and outline
        cv2.circle(image, center, 2, CIRCLE_CENTER_COLOR, 3)
        cv2.circle(image, center, radius, CIRCLE_OUTLINE_COLOR, 2)

        # Add number for debugging
        cv2.putText(
            image,
            str(i + 1),
            (center[0] - 10, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
else:
    print("No game circles detected")

# Display results
cv2.imshow("Detected Board and Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print summary
game_circle_count = len(circles[0]) if circles is not None else 0
print(f"\nSummary:")
print(f"Board detected: {'Yes' if board_circle is not None else 'No'}")
print(f"Expected game circles: 16")
print(f"Detected game circles: {game_circle_count}")

if game_circle_count != 16:
    print("\nTips for better detection:")
    print("- Adjust param2 (20-40) in detect_game_circles")
    print("- Modify the mask radius (0.85 factor)")
    print("- Check lighting and image quality")

# Save result
result = cv2.imwrite("assets/detected_board_and_circles.png", image)
if result:
    print("Image saved to assets/detected_board_and_circles.png")
else:
    print("Failed to save image")
