from ultralytics.data.annotator import auto_annotate
auto_annotate(data="match/images", det_model="models/yolov8x.pt", sam_model="models/sam_b.pt",output_dir="segmation")