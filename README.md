# Cyclical Curriculum Learning
This is an implementation of the paper: [Cyclical Curriculum Learning](https://ieeexplore.ieee.org/abstract/document/10103632), published in [IEEE Transactions on Neural Networks and Learning Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385).

## Example Usage

#### Train vanilla model
```python
vanilla_history = model.fit(X_train, y_train, 
                            validation_data = (X_test, y_test),
                            epochs = EPOCHS)
```

#### Obtain scores from trained model's losses
```python
losses = get_categ_ind_loss(model, X_train, y_train)
scores = 1 / losses
```

#### Train the new model with Cyclical Curriculum Learning
```python
cyclical_model, current_max, result_dict = CyclicalTrain(
    model=get_cifar10_model(),
    x=X_train,
    y=y_train,
    data_sizes=get_cycle_sizes(start_percent, end_percent, multiplier, EPOCHS),
    scores=scores,
    data=(X_test, y_test)
)
```

#### Get higher accuracy 😀
```python
# Log highest accuracy for vanilla and cyclical models
print("Vanilla Highest Accuracy:", round(max(vanilla_history.history['val_accuracy']),4)) # 0.7030
print("Cyclical Highest Accuracy:", round(max(result_dict['val_accuracy']),4))            # 0.7408

# Log highest 3 accuracies for vanilla and cyclical models
print("Vanilla Highest 3 Accuracies:", sorted(np.round(vanilla_history.history['val_accuracy'],4), reverse = True)[:3]) # [0.7030, 0.7027, 0.6999]
print("Cyclical Highest 3 Accuracies:", sorted(np.round(result_dict['val_accuracy'],4), reverse = True)[:3])            # [0.7408, 0.7407, 0.7404]
```

#### Full Code
Refer to [examples/cifar.ipynb](https://github.com/CyclicalCurriculum/Cyclical-Curriculum/blob/main/example/cifar.ipynb).

# Citations
```bibtex
@article{kesgin2023cyclical,
  title={Cyclical curriculum learning},
  author={Kesgin, H Toprak and Amasyali, M Fatih},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```

# License
MIT
