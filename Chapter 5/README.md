## Generalization
- The goal of machine learning is to perform well on **unseen data**
- Training performance alone does not indicate real-world usefulness
- Generalization measures how well learned patterns transfer beyond the training set

---

## Underfitting
- Occurs when the model is too simple to capture data structure
- Characteristics:
  - High training loss
  - High validation loss
- Common causes:
  - Insufficient model capacity
  - Poor optimization
  - Too few training epochs

---

## Overfitting
- Occurs when the model fits noise instead of signal
- Characteristics:
  - Low training loss
  - High validation loss
- More likely when:
  - Model capacity is high
  - Dataset is small
  - Input dimensionality is large

---

## Model Capacity
- Capacity refers to a model’s ability to fit complex functions
- Increasing capacity:
  - Always improves training performance
  - Eventually degrades validation performance
- There exists an optimal capacity between underfitting and overfitting

---

## Training vs Validation Behavior
- Training loss generally decreases monotonically
- Validation loss:
  - Decreases initially
  - Increases once overfitting begins
- The best model is often **not** the one with the lowest training loss

---

## High Dimensionality and Noise
- Real-world data uses only a small number of meaningful degrees of freedom
- Adding random but varying features:
  - Increases dimensionality
  - Introduces spurious correlations
  - Harms generalization
- Adding constant or non-varying features:
  - Does not affect generalization
  - Is easy for the model to ignore

---

## Manifold Hypothesis
- Real-world data lies on low-dimensional manifolds
- These manifolds are embedded in high-dimensional input spaces
- Neural networks learn transformations that:
  - Untangle data manifolds
  - Make classes separable
- Overfitting corresponds to poor behavior off the data manifold

---

## Evaluation Protocol
- **Training set**
  - Used to update model parameters
- **Validation set**
  - Used for hyperparameter tuning and model selection
  - Influences the model indirectly
- **Test set**
  - Used once at the end
  - Provides the least biased performance estimate
- Validation results are optimistic due to repeated tuning

---

## Baselines
- Always compare model performance to a simple baseline
- Examples:
  - Predicting the most frequent class
  - Random guessing (for balanced datasets)
- A model that does not outperform a baseline is not useful

---

## Evaluation Pitfalls
- Information leakage during preprocessing
- Non-representative data splits
- Incorrect metric choice (accuracy may be misleading)

---

## Optimization vs Model Design
- Poor optimization can mimic underfitting
- Common causes:
  - Learning rate too small → slow or stalled learning
  - Learning rate too large → unstable training
  - Too few epochs → incomplete convergence
- Optimization should be corrected before increasing model complexity

---

## Architecture Choice
- Using architectures aligned with data structure improves learning
- Examples:
  - Images → convolutional layers
  - Sequences → recurrent or attention-based models
- Proper inductive bias:
  - Reduces overfitting
  - Speeds up training

---

## Increasing Model Capacity
- Methods:
  - Adding more layers
  - Increasing units per layer
- Effects:
  - Improves training fit
  - Increases overfitting risk
- Requires stronger regularization

---

## Regularization

### Weight Decay (L2 Regularization)
- Penalizes large weights
- Encourages smoother functions
- Limits effective model complexity

---

### Dropout
- Randomly disables neurons during training
- Prevents co-adaptation
- Improves robustness and generalization

---

### Early Stopping
- Stops training when validation performance degrades
- Prevents memorization of noise
- Acts as implicit regularization

---

### Noise Injection
- Adds noise to inputs or activations during training
- Encourages stable and robust representations

---

## Dataset Quality
- More data improves generalization
- Clean labels matter more than raw quantity
- Training data should match real-world distributions

---

## Feature Engineering
- Still relevant in deep learning
- Removing irrelevant features improves generalization
- Reducing dimensionality reduces overfitting risk

---

## Core Takeaways
- Training performance does not measure success
- Overfitting is inevitable but controllable
- Evaluation protocol matters as much as model choice
- Fix optimization before architecture
- Regularization is essential for generalization
- Data quality often outweighs model complexity

