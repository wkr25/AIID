from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def Transform(image_size=64):
    return transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]
        )
    ])
#转换函数，将图片变成张量

def Dataloader(data_dir,image_size=64,batch_size=32,shuffle=True):
    transform=Transform(image_size)
    dataset=datasets.ImageFolder(root=data_dir,transform=transform)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader,dataset

if __name__=="__main__":
    train_loader,train_dataset=Dataloader(
        data_dir="data/train",
        image_size=64,
        batch_size=8,
        shuffle=True
    )
    print(train_dataset.class_to_idx)
    image,label=next(iter(train_loader))
    print(image.shape)
    print(label.shape)
    print(label)