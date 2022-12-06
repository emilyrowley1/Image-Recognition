#print("Hello World")

# from imageai.Prediction.Custom import ClassificationModelTrainer

# model_trainer = ClassificationModelTrainer()
# model_trainer.setModelTypeAsResNet50()
# model_trainer.setDataDirectory("idenprof")
# model_trainer.trainModel(num_objects=10, num_experiments=200, enhance_data=True, batch_size=32, show_network_summary=True)

#Installing the library
from imageai.Classification import ImageClassification
import os
execution_path = os.getcwd()
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
# the training for the image reconginition
prediction.setModelPath("resnet50_imagenet_tf.2.0.h5")
prediction.loadModel()

# predictions code, what image it is predicting
predictions, percentage_probabilities = prediction.classifyImage("blossom.jpg", result_count=5)
for index in range(len(predictions)):
    print(predictions[index] , " : " , percentage_probabilities[index])


# Video trainer
# from imageai.Detection.Custom import CustomVideoObjectDetection
# import os

# execution_path = os.getcwd()

# video_detector = CustomVideoObjectDetection()
# video_detector.setModelTypeAsYOLOv3()
# video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
# video_detector.setJsonPath("detection_config.json")
# video_detector.loadModel()

# video_detector.detectObjectsFromVideo(input_file_path="holo1.mp4",
#                                           output_file_path=os.path.join(execution_path, "holo1-detected3"),
#                                           frames_per_second=20,
#                                           minimum_percentage_probability=40,
#                                           log_progress=True)
