with_fork = False

train_joint = True

use_topological = False

use_subset = False

if with_fork:
    input_size = 17
else:
    input_size = 15

if use_topological:
    input_size = 18

if use_subset:
    input_size = 12

if train_joint:
    input_size = 29

batch_size = 64
epochs = 1000
val_split = 0.2
lr = 1e-4
decay = 1e-6
momentum = 0.9

train_count = 10000
test_count = 800