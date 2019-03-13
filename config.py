# Data configurations
path_name = 'Path'
class_names = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices']

uncertainty_mode = None
root_dir = '/home/suo/data/CheXpert-v1.0'
image_size = 224

num_workers = 10
batch_size = 64


# Model configurations
model_name = 'DenseNet'
optimizer_name = 'SGD'
lr = 1e-2
loss_name = 'BCELoss'

# Experiment configurations
epochs = 10
seed = 9999
disable_cuda = False
experiment_name = 'debug'