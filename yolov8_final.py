import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import csv
import os
from glob import glob
import requests
import datetime
import pytz

# ========================= TensorRT Engine Handler =========================
class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Get input/output info and allocate GPU buffers
        self.input_name = None
        self.input_shape = None
        self.input_dtype = None
        self.output_names = []
        self.bindings = []        # device pointers
        self.binding_addrs = []   # int addresses passed to execute_v2

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            size = int(trt.volume(shape))

            # Allocate GPU memory
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.bindings.append(device_mem)
            self.binding_addrs.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.input_name = name
                self.input_shape = tuple(shape)
                self.input_dtype = dtype
            else:
                self.output_names.append(name)

        print("[INFO] Engine loaded successfully!")
        print(f"[INFO] Input: {self.input_name} - Shape: {self.input_shape}")
        # print output shapes (best-effort)
        for idx, name in enumerate(self.output_names):
            # compute index for binding shape retrieval
            binding_index = len(self.bindings) - len(self.output_names) + idx
            try:
                shape = self.engine.get_binding_shape(binding_index)
            except:
                shape = "unknown"
            print(f"[INFO] Output {idx}: {name} - Shape: {shape}")

    def infer(self, input_data):
        """Run inference with input_data: numpy array (batch, C, H, W)"""
        # ensure contiguous and correct dtype
        inp = np.ascontiguousarray(input_data.astype(np.float32))
        # copy to GPU (first binding is input)
        cuda.memcpy_htod(self.bindings[0], inp)

        # run
        self.context.execute_v2(bindings=self.binding_addrs)

        # copy outputs back
        outputs = []
        # note: bindings[0] is input; outputs start from 1
        for i in range(1, len(self.bindings)):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            out = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(out, self.bindings[i])
            outputs.append(out)
        return outputs


# ========================= Image Preprocessing =========================
def preprocess_image(image_or_path, input_size=640):
    """
    Hỗ trợ cả:
      - image_or_path: đường dẫn file (str)
      - image_or_path: numpy.ndarray (OpenCV BGR image)
    Trả về:
      img_batch: shape (1, C, H, W), dtype=float32 (RGB normalized)
      orig_img: ảnh gốc (BGR uint8)
      scale_info: (scale, pad_left, pad_top) để scale_coords sử dụng
    """
    # load image (path or array)
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        img = image_or_path.copy()
    else:
        raise ValueError("preprocess_image expects str path or numpy.ndarray")

    orig_h, orig_w = img.shape[:2]

    # letterbox resize (keep aspect ratio)
    scale = min(input_size / orig_h, input_size / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    if new_h == 0 or new_w == 0:
        raise ValueError("Input image too small or invalid dimensions")

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # pad to square input_size x input_size with value 114
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_top = (input_size - new_h) // 2
    pad_left = (input_size - new_w) // 2
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

    # BGR -> RGB, normalize to [0,1]
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0

    # CHW and add batch dimension
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)
    img_batch = np.ascontiguousarray(img_batch)

    return img_batch, img, (scale, pad_left, pad_top)


# ========================= Post-processing (OPTIMIZED) =========================
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def nms(boxes, scores, iou_threshold=0.7):
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def process_output_optimized(output0, output1, conf_thresh=0.25, iou_thresh=0.7):
    """
    Phiên bản tối ưu của process_output.
    
    Args:
        output0: [1, 38, N] - detection output
        output1: [1, 32, H, W] - proto masks
    
    Returns:
        boxes, scores, classes, mask_coefs (tất cả TRƯỚC KHI scale về ảnh gốc)
    """
    out0 = output0[0].transpose(1, 0)  # [N, 38]
    
    boxes = out0[:, :4]            # [N, 4] xywh
    class_scores = out0[:, 4:6]    # [N, 2] class scores
    mask_coefs = out0[:, 6:]       # [N, 32]
    
    # Lấy max scores và class ids
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # Lọc theo confidence
    keep_mask = max_scores > conf_thresh
    boxes = boxes[keep_mask]
    scores = max_scores[keep_mask]
    classes = class_ids[keep_mask]
    mask_coefs = mask_coefs[keep_mask]
    
    if boxes.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Chuyển xywh sang xyxy
    boxes = xywh2xyxy(boxes)
    
    # NMS
    keep_idx = nms(boxes, scores, iou_thresh)
    if len(keep_idx) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    classes = classes[keep_idx]
    mask_coefs = mask_coefs[keep_idx]
    
    return boxes, scores, classes, mask_coefs


def crop_mask(masks, boxes):
    """
    Crop masks bằng cách zero out mọi thứ bên ngoài bbox.
    Vectorized implementation như Ultralytics.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes, 4, axis=1)  # [n, 1] mỗi cái
    
    # Tạo coordinate grids
    r = np.arange(w, dtype=np.float32)[None, None, :]  # [1, 1, w]
    c = np.arange(h, dtype=np.float32)[None, :, None]  # [1, h, 1]
    
    # Tạo crop mask: True bên trong bbox, False bên ngoài
    crop_mask = (r >= x1[:, :, None]) & (r < x2[:, :, None]) & \
                (c >= y1[:, None, :]) & (c < y2[:, None, :])
    
    # Áp dụng crop
    return masks * crop_mask


def process_mask_ultralytics(protos, masks_in, bboxes, shape):
    """
    Xử lý masks theo đúng cách Ultralytics.
    
    Args:
        protos: [32, H, W] - proto masks từ model
        masks_in: [n, 32] - mask coefficients cho n detections
        bboxes: [n, 4] - bounding boxes ở định dạng xyxy (trong không gian input_size)
        shape: (H, W) - kích thước output mục tiêu (kích thước ảnh input_size)
    
    Returns:
        masks: [n, H, W] - binary masks ở kích thước input_size
    """
    c, mh, mw = protos.shape  # 32, H, W
    ih, iw = shape  # input_size dimensions
    
    # Matrix multiplication: masks_in @ protos -> [n, H, W]
    masks = np.matmul(masks_in, protos.reshape(c, -1)).reshape(-1, mh, mw)
    
    # Sigmoid activation
    masks = 1.0 / (1.0 + np.exp(-masks))
    
    # Downsample bboxes về proto mask resolution
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / ih  # x1
    downsampled_bboxes[:, 2] *= mw / ih  # x2
    downsampled_bboxes[:, 1] *= mh / iw  # y1
    downsampled_bboxes[:, 3] *= mh / iw  # y2
    
    # Crop masks về bounding boxes trong proto space
    masks = crop_mask(masks, downsampled_bboxes)
    
    # Upsample về kích thước input_size
    masks_upsampled = []
    for mask in masks:
        mask_resized = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR)
        masks_upsampled.append(mask_resized)
    
    masks_upsampled = np.array(masks_upsampled)
    
    # Binary threshold
    masks_binary = (masks_upsampled > 0.5).astype(np.uint8)
    
    return masks_binary


def scale_boxes(boxes, input_shape, orig_shape, scale_info):
    """
    Scale boxes từ input image (với padding) về kích thước ảnh gốc.
    
    Args:
        boxes: [n, 4] - boxes ở định dạng xyxy (trong không gian input_size)
        input_shape: (input_size, input_size)
        orig_shape: (orig_h, orig_w)
        scale_info: (scale, pad_left, pad_top)
    
    Returns:
        boxes: [n, 4] - boxes đã scale về ảnh gốc
    """
    if boxes.size == 0:
        return boxes
    
    scale, pad_left, pad_top = scale_info
    boxes = boxes.copy().astype(float)
    
    # Loại bỏ padding
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    
    # Scale về kích thước gốc
    boxes[:, [0, 2]] /= scale
    boxes[:, [1, 3]] /= scale
    
    # Clip về bounds của ảnh
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
    
    return boxes.astype(int)


def generate_masks_optimized(mask_coefs, proto_masks, boxes, orig_shape, scale_info, 
                            input_size=640, mask_threshold=0.5):
    """
    Hàm chính để generate masks - phiên bản tối ưu.
    
    Args:
        mask_coefs: [n, 32] - mask coefficients sau NMS
        proto_masks: [1, 32, H, W] - proto masks từ model
        boxes: [n, 4] - boxes trong không gian input_size (sau NMS, trước scaling)
        orig_shape: (orig_h, orig_w) - kích thước ảnh gốc
        scale_info: (scale, pad_left, pad_top) - thông tin letterbox
        input_size: kích thước input của model (640)
        mask_threshold: ngưỡng để tạo binary mask
    
    Returns:
        masks: list của [orig_h, orig_w] binary masks
        boxes_scaled: [n, 4] - boxes đã scale về ảnh gốc
    """
    if mask_coefs.size == 0:
        return [], np.array([])
    
    # Scale boxes về ảnh gốc trước
    boxes_scaled = scale_boxes(boxes, (input_size, input_size), orig_shape, scale_info)
    
    # Xử lý masks trong không gian input_size
    proto = proto_masks[0]  # [32, H, W]
    
    # Generate masks sử dụng phương pháp Ultralytics
    masks_input_size = process_mask_ultralytics(
        proto, 
        mask_coefs, 
        boxes,  # boxes trong không gian input_size
        (input_size, input_size)  # xử lý trong không gian input_size trước
    )
    
    # Scale masks về kích thước ảnh gốc
    masks_final = []
    scale, pad_left, pad_top = scale_info
    
    for mask_input in masks_input_size:
        # Loại bỏ padding từ mask
        pad_top_int = int(pad_top)
        pad_left_int = int(pad_left)
        
        # Tính kích thước unpadded
        unpad_h = int(input_size - 2 * pad_top)
        unpad_w = int(input_size - 2 * pad_left)
        
        # Crop out padding
        mask_unpad = mask_input[pad_top_int:pad_top_int+unpad_h, 
                                pad_left_int:pad_left_int+unpad_w]
        
        # Resize về kích thước ảnh gốc
        mask_orig = cv2.resize(mask_unpad, (orig_shape[1], orig_shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
        
        # Đảm bảo binary với threshold tùy chỉnh
        mask_orig = (mask_orig > mask_threshold).astype(np.uint8)
        
        masks_final.append(mask_orig)
    
    return masks_final, boxes_scaled


# ========================= Centroid Calculations =========================
def calculate_centroid(mask):
    if mask is None or mask.sum() == 0:
        return None, None
    mask_uint8 = (mask * 255).astype(np.uint8)
    M = cv2.moments(mask_uint8)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    return None, None

def calculate_kmeans_centroids(mask, k=3):
    """
    Tính K điểm trọng tâm (centroid) bằng KMeans (OpenCV) cho vùng mask (cỏ dại).
    Trả về list các (cx, cy).
    """
    if mask is None or mask.sum() == 0:
        return []

    # Lấy toàn bộ tọa độ pixel = 1 trong mask
    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys)).astype(np.float32)  # OpenCV yêu cầu float32

    if len(points) < k:
        # Nếu điểm ít hơn số cụm mong muốn thì chỉ trả về trung bình
        mean_x = int(points[:,0].mean())
        mean_y = int(points[:,1].mean())
        return [(mean_x, mean_y)]

    # Cài đặt tiêu chí dừng: max_iter hoặc epsilon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Áp dụng KMeans với k-means++ (cv2.KMEANS_PP_CENTERS)
    ret, labels, centers = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Đổi sang int (tọa độ pixel)
    centroids = [(int(x), int(y)) for x, y in centers]
    return centroids


# ========================= Visualization =========================
def draw_results(image, boxes, scores, classes, masks, centroids,
                 class_names=['crop', 'weed'],
                 crop_color=(144, 238, 144),  # light green (B,G,R)
                 weed_color=(203, 192, 255),  # light pink-ish (B,G,R)
                 alpha=0.4):
    """
    Draw masks (light colors), bounding boxes.
    Label confidence for both classes.
    """
    result = image.copy()
    h, w = result.shape[:2]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls = int(classes[i])
        score = float(scores[i])
        mask = masks[i] if i < len(masks) else np.zeros((h, w), dtype=np.uint8)

        # choose color by class
        if cls == 1:  # weed
            color = weed_color
        else:         # crop or others
            color = crop_color

        # create colored mask image
        if mask is not None and mask.sum() > 0:
            mask_3ch = np.stack([mask]*3, axis=-1).astype(np.uint8)
            colored_mask = (mask_3ch * np.array(color, dtype=np.uint8))
            # overlay with alpha
            result = cv2.addWeighted(result, 1.0, colored_mask, alpha, 0)

        # draw bbox (use same color but slightly stronger)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # draw label
        label = f"{class_names[cls]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # filled background for text
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # draw centroids (red dots) for weed class only
        if cls == 1 and i < len(centroids):
            centroid_list = centroids[i]
            for (cx, cy) in centroid_list:
                cv2.circle(result, (cx, cy), 2, (0, 0, 255), -1)

    return result


# ========================= Configuration =========================
MODEL_PATH = './src/best_final.engine'

print("=" * 60)
print("Loading TensorRT Engine...")
print("=" * 60)
engine = TRTEngine(MODEL_PATH)

CONFIG = {
    'INPUT_SIZE': 640,
    'CONF_THRESH': 0.5,
    'IOU_THRESH': 0.7,
    'WEED_CLASS_ID': 1,    
    'CLASS_NAMES': ['crop', 'weed']
}


# ========================= Main Processing Function =========================
def process_single_image(img_input, engine=engine, **kwargs):
    """
    Xử lý một hình ảnh qua mô hình AI, trả về ảnh kết quả và danh sách tọa độ tâm của cỏ dại.
    
    Args:
        img_input: Đường dẫn ảnh (str) hoặc numpy array
        engine: TRTEngine instance
        **kwargs: Config override (CONF_THRESH, IOU_THRESH, etc.)
    
    Returns:
        result_img: Ảnh đã vẽ kết quả
        weed_centroids: List các tọa độ (cx, cy) của cỏ dại
        weed_count: Số lượng cỏ dại phát hiện
    """
    cfg = {**CONFIG, **kwargs}  # Gộp config mặc định và config truyền vào

    INPUT_SIZE = cfg['INPUT_SIZE']
    CONF_THRESH = cfg['CONF_THRESH']
    IOU_THRESH = cfg['IOU_THRESH']
    WEED_CLASS_ID = cfg['WEED_CLASS_ID']
    CLASS_NAMES = cfg['CLASS_NAMES']
    weed_count = 0

    # --- 1. Tiền xử lý ---
    try:
        if isinstance(img_input, str):
            input_data, orig_img, scale_info = preprocess_image(img_input, INPUT_SIZE)
            file_name = os.path.basename(img_input)
        else:
            input_data, orig_img, scale_info = preprocess_image(img_input, INPUT_SIZE)
            file_name = "live_frame"
    except Exception as e:
        print(f"Lỗi tiền xử lý: {e}")
        return img_input if isinstance(img_input, np.ndarray) else None, [], 0

    # --- 2. Suy luận ---
    outputs = engine.infer(input_data)
    if len(outputs) < 2:
        print("-> Lỗi: Đầu ra từ engine không đủ.")
        return orig_img, [], 0

    output0, output1 = outputs[1], outputs[0]

    # --- 3. Hậu xử lý với hàm optimized ---
    boxes, scores, classes, mask_coefs = process_output_optimized(
        output0, output1, 
        conf_thresh=CONF_THRESH,
        iou_thresh=IOU_THRESH
    )
    
    if boxes.size == 0:
        return orig_img, [], 0

    # --- 4. Tạo mask với generate_masks_optimized ---
    orig_h, orig_w = orig_img.shape[:2]
    
    masks_final, boxes_scaled = generate_masks_optimized(
        mask_coefs, 
        output1, 
        boxes,  # boxes trong không gian INPUT_SIZE (trước scaling)
        (orig_h, orig_w),  # kích thước ảnh gốc
        scale_info,  # (scale, pad_left, pad_top)
        input_size=INPUT_SIZE,
        mask_threshold=0.5  # Có thể điều chỉnh threshold
    )
    
    if len(masks_final) == 0:
        return orig_img, [], 0

    # --- 5. Tính toán centroids (masks đã ở kích thước ảnh gốc) ---
    centroids, weed_centroids = [], []
    
    for i, mask_final in enumerate(masks_final):
        cls_id = int(classes[i])
        
        # Tính K-means centroids (k=1 như code gốc)
        centroid_list = calculate_kmeans_centroids(mask_final, k=1)
        centroids.append(centroid_list)
        
        # Chỉ lưu centroids của weed
        if cls_id == WEED_CLASS_ID:
            weed_count += 1
            weed_centroids.extend(centroid_list)

    # --- 6. Vẽ kết quả ---
    result_img = draw_results(
        orig_img, boxes_scaled, scores, classes,
        masks_final, centroids, class_names=CLASS_NAMES
    )

    return result_img, weed_centroids, weed_count