import torchreid

data_manager = torchreid.data.ImageDataManager(
    root='/Users/DrMatters/Documents/git/vision/data/datasets/',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop'],
)
dataloaders = data_manager.return_dataloaders()
lo1 = dataloaders[0]
first = next(enumerate(lo1))[1]

pass
