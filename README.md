# Custom Training Loop using TensorFlow Keras

In this repository, we describe the steps for creating a custom training loop using TensorFlow Keras for a supervised learning task. Specifically, we train a vision deep learning model (VGGNet) for multi-class classification of the CIFAR10 dataset. A custom training loop offers flexibility and granular-level control over the training process. 

- Notebook 1: single-device training (e.g., single GPU/CPU). 

- Notebook 2: distributed training (single-host, multi-device).


### Custom Training Loop: Tasks
The main task in a custom training loop is to compute the loss gradients by differentiating the loss function with respect to the model parameters (weights), which is followed by updating the weights. The differentiation is performed automatically by the tf.GradientTape API. During the two stages of training (forward and backward passes), the following sub-tasks are performed by the functions of this API.

- Forward pass: Remember what operations (that involve a variable) happen in what order during the forward pass. This is done by tf.GradientTape to "record" relevant operations executed inside the context of a tf.GradientTape onto a "tape".
- Backward pass: Use the "tape" to traverse the "recorded" list of operations in reverse order to compute gradients (i.e., reverse mode differentiation).


### A Quick Recipe for Creating a Custom Training Loop

- Create two nested loops: one for the epochs, and the other for the batches within an epoch.

- Within the inner loop (for iterating through the batches within an epoch):

      Forward pass:
      -- Define the tf.GradientTape() block. 
      -- Inside the block, make a prediction for one batch (using the model as a function), and compute the loss.
      The loss consists of the main loss plus the other losses (e.g., weight regularizer loss). 
      Note that, to save memory, only put the strict minimum operations inside the tf.GradientTape() block. The tape is automatically erased immediately after its gradient() method is executed.

      Backward pass:
      -- Ask the tape to compute the gradient of the loss with respect to each trainable variable (by using the gradient() method).

      -- Apply the gradients to the optimizer to perform a Gradient Descent step (by using the apply_gradient() method).

      -- Update the mean loss and the metrics (over the current epoch).

      -- Display the status bar to show the training statistics for each iteration and/or epoch.


More information on creating custom training loops: https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough


## Summary of the Other Techniques/Tools Used in this Notebook

In addition to describing the design of a custom training loop, we use the following techniques/tools that are useful in practical deep learning tasks.

- Store and pre-process the data in the TensorFlow Dataset format.

- Build a model by defining a custom layer.

- Utilize various optimizers and schedulers.

- Model serialization.

        -- Serialize the final model (with custom layers) in TensorFlow's SavedModel format. 
        -- Serialize the intermediate model checkpoints (only the parameters of the model) using the tf.train.Checkpoint class. Customize the checkpoint object.

- Loading the saved model.
        
        -- Final SavedModel
        -- Intermediate model checkpoints

- Monitor the Training Process

        -- Use comet ml for monitoring the training in real time.
   
