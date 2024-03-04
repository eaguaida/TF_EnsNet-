# Import necessary libraries
import numpy as np
import tensorflow as tf

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, val_dataset):
        score = model.evaluate(val_dataset, verbose=0)
        print(f'Test loss: {score[0]}')
        print(f'Test accuracy: {score[1]}')
    
    @staticmethod
    def predict_and_vote(model, test_dataset):
        predictions = model.predict(test_dataset)
        return ModelEvaluator.manual_majority_vote(predictions)

    @staticmethod
    def manual_majority_vote(predictions):
        # Assuming predictions shape is (num_subnets, num_samples, num_classes)
        # We first transpose to get shape (num_samples, num_subnets, num_classes)
        predictions = np.transpose(predictions, (1, 0, 2))
        
        # Now we get the class with the highest probability from each subnet
        votes = np.argmax(predictions, axis=-1)  # Shape becomes (num_samples, num_subnets)
        
        # Apply majority voting across subnets
        final_predictions = np.array([np.bincount(votes[i]).argmax() for i in range(votes.shape[0])])
        return final_predictions
