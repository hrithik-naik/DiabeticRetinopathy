import tensorflow as tf
import segmentation_models as sm
import cv2
import numpy as np
import matplotlib.pyplot as plt

class InferencePipeline:
    def __init__(self, model_path, backbone='vgg16', input_shape=(512, 512, 1)):
        self.backbone = backbone
        self.input_shape = input_shape
        self.model = self.load_model(model_path)
        

    def load_model(self, model_path):
        n_classes = 1
        activation = 'sigmoid'
        optim_gray = tf.keras.optimizers.Adam(0.0001)

        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
        total_loss_gray = 0.9 * dice_loss + 0.1 * focal_loss

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        model = sm.Unet(
            self.backbone,
            input_shape=self.input_shape,
            encoder_weights=None,
            classes=n_classes,
            activation=activation
        )
        model.compile(optimizer=optim_gray, loss=total_loss_gray, metrics=metrics)
        model.load_weights(model_path)
        return model

    def load_img(self, addr):
        img = cv2.imread(addr, 0)
        img = img.astype('float32')
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_AREA)
        img = img.reshape(self.input_shape)
        return img

    def predict(self, image_path):
        img = self.load_img(image_path)
        prediction = self.model.predict(np.expand_dims(img, axis=0))
        return prediction

    def threshold_fundus_image(self, fundus_image):
        # Thresholding the fundus image to isolate the fundus area
        _, thresholded = cv2.threshold(fundus_image, 50, 255, cv2.THRESH_BINARY_INV)
        return thresholded

    def calculate_areas(self, thresholded_fundus, mask):
        # Calculate areas of the fundus and the mask
        fundus_area = np.sum(thresholded_fundus > 0)
        masked_area = np.sum(mask > 0)
        ratio = fundus_area / masked_area if fundus_area > 0 else 0
        return fundus_area, masked_area, ratio
    



    def process_image(self, image_path):
        original_image = cv2.imread(image_path, 0)
        original_image = cv2.resize(original_image, (self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_AREA)
        
       
        thresholded_fundus = self.threshold_fundus_image(original_image)

       
        prediction_mask = self.predict(image_path)

       
        fundus_area, masked_area,ratio = self.calculate_areas(thresholded_fundus, prediction_mask)
        return fundus_area,masked_area,ratio
        
 



