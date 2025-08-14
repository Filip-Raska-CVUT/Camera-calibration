import cv2
import numpy as np
import math
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

def detect_corners_findContours(png_folder):
    show_image = True
    
    for file in os.listdir(png_folder):
        if not file.endswith('.png'):
            continue

        img_path = os.path.join(png_folder, file)
        orig = cv2.imread(img_path)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        # Invert threshold so black shapes are foreground (findContours sees holes).
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            print("No contours found")
            continue

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

def detect_corners_floodfill(png_folder):
    show_image = True

    for file in os.listdir(png_folder):
        if not file.endswith('.png'):
            continue

        img_path = os.path.join(png_folder, file)
        orig = cv2.imread(img_path)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        # Binary inverted so white shape is foreground
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        h, w = binary.shape

        # TODO debug
        wh = (w // 2, h // 2)

        mask = np.zeros((h+2, w+2), np.uint8)  # required by floodFill

        center_point = (w // 2, h // 2)

        # Make a copy so floodFill doesn’t overwrite original binary
        flood_filled = binary.copy()

        # Flood-fill from center
        cv2.floodFill(flood_filled, mask, center_point, 128)

        # Extract mask of the filled region
        filled_region = (mask[1:-1, 1:-1] != 0).astype(np.uint8) * 255

        # Find contour of that region
        contours, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = orig.copy()
        ANGLE_THRESHOLD = 130

        # the corners are in no particular order set allows set difference to compare with the points generated from id
        rel_corner_coord = set()

        for cnt in contours:
            eps = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            points = [(int(p[0][0]), int(p[0][1])) for p in approx]

            for j, (px, py) in enumerate(points):
                prev_point = points[j - 1]
                next_point = points[(j + 1) % len(points)]
                ang = angle(prev_point, (px, py), next_point)

                if ang < ANGLE_THRESHOLD:
                    cv2.circle(result, (px, py), 5, (0, 0, 255), -1)

                    # relative coordinates with respect to the marker center
                    # x is to the left and y is up
                    rel_corner_coord.add((px - center_point[0], center_point[1] - py))

        # TODO debug
        if show_image:
            """
            cv2.imshow("FloodFill center corners", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            print(rel_corner_coord)

            # test
            return wh, result

def polar_to_cartesian(r, theta, wh):
    theta_rad = math.radians(theta)
    x = math.ceil(r * math.cos(theta_rad)) + wh[0]
    y = math.ceil(r * math.sin(theta_rad)) + wh[1]

    return (x, y)

def draw_whycode_marker(id, teethCount, r, wh, result):
    points = set()
    # for each tooth we need to make 2 smaller teeth
    angle_step = 360 / teethCount / 2 

    # converts id to binary
    binary = format(id, '032b')[-teethCount:]
    # binary = binary[-teethCount:]
    print(binary)

    # the current angle
    theta = 0

    # radiues adjuster
    alpha = 0.6

    for step in range(teethCount):
        print(binary[step])

        if binary[step] == '1':
            p1 = polar_to_cartesian(r * alpha, theta, wh)
            theta += angle_step
            p2 = polar_to_cartesian(r * alpha, theta, wh)
            p3 = polar_to_cartesian(r, theta, wh)
            theta += angle_step
            p4 = polar_to_cartesian(r, theta, wh)
        else:
            p1 = polar_to_cartesian(r, theta, wh)
            theta += angle_step
            p2 = polar_to_cartesian(r, theta, wh)
            p3 = polar_to_cartesian(r * alpha, theta, wh)
            theta += angle_step
            p4 = polar_to_cartesian(r * alpha, theta, wh)

        points.add(p1)
        points.add(p2)
        points.add(p3)
        points.add(p4)

        cv2.circle(result, (p1[0], p1[1]), 5, (100, 150, 0), -1)
        cv2.circle(result, (p2[0], p2[1]), 5, (100, 150, 0), -1)
        cv2.circle(result, (p3[0], p3[1]), 5, (100, 150, 0), -1)
        cv2.circle(result, (p4[0], p4[1]), 5, (100, 150, 0), -1)

    print(points)
    cv2.imshow("FloodFill center corners", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points

convert_svg_to_png('6bit', '6bit_png')
# detect_corners_findContours('6bit_png')
wh, result = detect_corners_floodfill('6bit_png')
draw_whycode_marker(62, 6, 180, wh, result)