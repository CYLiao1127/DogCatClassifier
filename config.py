class Config:
    model_name = 'resnet18'
    data_dir = './data/data'
    img_size = 224
    batch_size = 128
    epochs = 10
    lr = 1e-4
    num_classes = 2
    device = 'cuda'
    result_save_path = 'result'
    model_save_path = './result.pth'