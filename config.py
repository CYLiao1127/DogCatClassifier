class Config:
    data_dir = './data/data'
    img_size = 224
    batch_size = 8
    epochs = 20
    lr = 1e-4
    num_classes = 2
    device = 'cuda'
    model_save_path = './result.pth'