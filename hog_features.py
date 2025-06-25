# HOG features
def video_to_hog_sequence(video_path, size=(90, 180)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    features = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and convert to grayscale
        frame = cv2.resize(frame, (size[1], size[0]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use YOLO to detect people
        results = model(gray)
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        person_boxes = [box for box in bboxes if int(results[0].boxes.cls[0]) == 0]  # class 0 = person

        if prev_gray is not None and len(person_boxes) >= 2:
            # Find max intersection box pair
            max_inter_box = get_max_intersection_box(person_boxes)
            if max_inter_box is not None:
                x1, y1, x2, y2 = max_inter_box
                crop_prev = prev_gray[y1:y2, x1:x2]
                crop_now = gray[y1:y2, x1:x2]

                # Resize cropped regions
                crop_prev = cv2.resize(crop_prev, (64, 64))
                crop_now = cv2.resize(crop_now, (64, 64))

                # Compute optical flow and HOG
                flow = compute_optical_flow(crop_prev, crop_now)
                hog = extract_hog(flow)
                features.append(hog)  # Already 1D

        prev_gray = gray

    cap.release()

    if len(features) >= seq_len:
        features = features[:seq_len]
    else:
        # Pad with zeros if less than seq_len
        features += [np.zeros_like(features[0])] * (seq_len - len(features))

    return np.array(features)
#5
VIDEO_DIR = '/content/drive/MyDrive/VF_246'
X, y = [], []
for fname in os.listdir(VIDEO_DIR):
    if not fname.lower().endswith('.avi'): continue
    label = 0 if 'nf' in fname.lower() else 1
    video_path = os.path.join(VIDEO_DIR, fname)
    seq = video_to_hog_sequence(video_path)
    #seq = video_to_hog_sequence(os.path.join(VIDEO_DIR, fname))
    if seq is not None and len(seq) == 16:
        X.append(seq)
        y.append(label)

X = np.array(X)  # shape = (N, seq_len, hog_dim)
y = np.array(y)
print("Samples:", X.shape)
