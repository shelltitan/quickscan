import transformations
import numpy as np

x = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
y = np.random.randint(10, 15, size=(128, 128), dtype=np.uint8)

transforms = transformations.Compose([
    transformations.Resize(input_size=(64, 64, 3), target_size=(64, 64)),
    transformations.DenseTarget(),
    transformations.MoveAxis(),
    transformations.Normalize01()
])

x_t, y_t = transforms(x, y)

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'x_t: shape: {x_t.shape}  type: {x_t.dtype}')
print(f'x_t = min: {x_t.min()}; max: {x_t.max()}')

print(f'y = shape: {y.shape}; class: {np.unique(y)}')
print(f'y_t = shape: {y_t.shape}; class: {np.unique(y_t)}')