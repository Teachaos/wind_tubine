from ultralytics import SAM, YOLO, FastSAM

# Profile SAM-b
# model = SAM("sam_b.pt")
# model.info()
# model("ultralytics/assets")
#
# # Profile MobileSAM
# model = SAM("mobile_sam.pt")
# model.info()
# model("ultralytics/assets")

# Profile FastSAM-s
model = FastSAM("models/FastSAM-x.pt")
model.info()
results = model("match/images/DSC01139.jpg",conf=0.9)
results[0].show()

# # Profile YOLOv8n-seg
# model = YOLO("yolov8n-seg.pt")
# model.info()
# model("ultralytics/assets")