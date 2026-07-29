import numpy as np
import os

base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'UCI HAR Dataset')
files = {
    'train/X_train.txt':       (7352, 561),
    'train/y_train.txt':       (7352,),
    'train/subject_train.txt': (7352,),
    'test/X_test.txt':         (2947, 561),
    'test/y_test.txt':         (2947,),
    'test/subject_test.txt':   (2947,),
}

print('=== File Existence Check ===')
for f, expected in files.items():
    path = os.path.join(base, f)
    exists = os.path.exists(path)
    print(f'  {f}: exists={exists}, expected shape={expected}')

print()
print('=== Quick Data Check ===')
X = np.loadtxt(os.path.join(base, 'train/X_train.txt'))
y = np.loadtxt(os.path.join(base, 'train/y_train.txt'), dtype=int)
s = np.loadtxt(os.path.join(base, 'train/subject_train.txt'), dtype=int)

print(f'X shape:   {X.shape}   (expected (7352, 561))')
print(f'y shape:   {y.shape}   (expected (7352,))')
print(f's shape:   {s.shape}   (expected (7352,))')
print(f'y unique:  {np.unique(y)}  (expected [1 2 3 4 5 6])')
print(f's unique:  {np.unique(s)}')
print(f'X min/max: {X.min():.4f} / {X.max():.4f}  (expected ~-1 to 1)')
print()
print('All checks passed!' if X.shape == (7352, 561) and list(np.unique(y)) == [1,2,3,4,5,6] else 'WARNING: some checks failed.')
