import tensorflow as tf
from tensorflow.keras.models import Model

class FaceRecognizer(Model):
    def __init__(self, faceRec,  **kwargs): 
        super().__init__(**kwargs)
        self.model = faceRec

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        X, y = batch
        
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            y_r = tf.reshape(y[0], (10, 1))

            batch_classloss = self.closs(y_r, classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"val_total_loss": total_loss, "val_class_loss": batch_classloss, "val_regress_loss": batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        y_r = tf.reshape(y[0], (10, 1))

        batch_classloss = self.closs(y_r, classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"val_total_loss": total_loss, "val_class_loss": batch_classloss, "val_regress_loss": batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)