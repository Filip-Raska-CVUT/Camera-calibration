import cv2
import numpy as np
import cairosvg
import os

def convert_svg_to_png(svg_folder, png_folder):
    os.makedirs(png_folder, exist_ok=True)
    for file in os.listdir(svg_folder):
        if file.endswith('.svg'):
            svg_path = os.path.join(svg_folder, file)
            png_path = os.path.join(png_folder, file[:-4] + '.png')
            cairosvg.svg2png(url=svg_path, write_to=png_path)

def angle(pt1, pt2, pt3):
    # Compute angle between 3 points
    a = np.array(pt1)
    b = np.array(pt2)
    c = np.array(pt3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180 / np.pi

def detect_edges_debug(png_folder):
    show_image = True
    
    for file in os.listdir(png_folder):
        if not file.endswith('.png'):
            continue

        img_path = os.path.join(png_folder, file)
        orig = cv2.imread(img_path)
        img = orig.copy()
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        # Invert threshold so black shapes are foreground (findContours sees holes).
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            print("No contours found")
            continue

        """
        # --- DRAW ALL CONTOURS + indices so we can visually debug hierarchy
        debug = orig.copy()
        rng = np.random.default_rng(12345)
        for i, cnt in enumerate(contours):
            color = tuple(int(c) for c in (rng.integers(50,230), rng.integers(50,230), rng.integers(50,230)))
            cv2.drawContours(debug, contours, i, color, 2)

            # compute centroid for label
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                parent = int(hierarchy[0][i][3])

        if show_image:
            cv2.imshow("Contours", debug)
            cv2.waitKey(0)
        """

        # detecting corners
        result = orig.copy()
        center = np.array(orig.shape[1::-1]) / 2
        diagonal = np.linalg.norm(center)

        MAX_DIST = diagonal * 0.5
        ANGLE_THRESHOLD = 130  # degrees

        for i, cnt in enumerate(contours):
            parent = int(hierarchy[0][i][3])

            # Focus on child contours (holes) — these are likely the inner white circle boundaries.
            if parent == -1:
                continue

            if cv2.contourArea(cnt) < 30:
                continue

            # 1) approxPolyDP with SMALL epsilon to preserve sharp junctions
            eps = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            points = [(int(p[0][0]), int(p[0][1])) for p in approx]

            for j, (x, y) in enumerate(points):
                prev_point = points[j - 1]
                next_point = points[(j + 1) % len(points)]

                ang = angle(prev_point, (x, y), next_point)

                dist = np.linalg.norm(np.array([x, y]) - center)

                if dist < MAX_DIST and ang < ANGLE_THRESHOLD: 
                    cv2.circle(result, (x, y), 5, (0, 0, 255), -1)


        if show_image:
            cv2.imshow("Inner-contour corners", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # show_image = False

convert_svg_to_png('6bit', '6bit_png')
detect_edges_debug('6bit_png')

