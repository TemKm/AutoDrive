def limit_region(self, img_edges):
    height, width = img_edges.shape[:2]
    mask = np.zeros_like(img_edges)

    vertices = np.array([[(0, height), 
                          (width / 2, height / 2), 
                          (width, height)]], 
                          dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(img_edges, mask)

    # Convert to grayscale and then binary
    gray = cv2.cvtColor(masked_edges, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return binary
