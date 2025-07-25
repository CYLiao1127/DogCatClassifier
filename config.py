class Config:
    model_name = 'resnet18'
    data_dir = './data/data'
    best_model_path = './result/best_model.pth'
    img_size = 224
    batch_size = 128
    epochs = 10
    lr = 1e-4
    num_classes = 2
    device = 'cuda'
    result_save_path = 'result'
    use_mixup = False