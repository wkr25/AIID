import torch
from torch import nn
from model import Detector
from dataset import Dataloader

def train(model,train_loader,loss_fn,optimizer,device):
    model.train()
    total_loss=0
    total_correct=0
    total_samples=0

    for image,label in train_loader:
        image=image.to(device)
        label=label.to(device)
        logit,embedding=model(image)
        
        loss=loss_fn(logit,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss+=loss.item()*image.size(0)
        prediction=torch.argmax(logit,dim=1)
        total_correct+=(prediction==label).sum().item()
        total_samples+=image.size(0)

    avg_loss=total_loss/total_samples
    accuracy=total_correct/total_samples
    return avg_loss,accuracy

def test(model,test_loader,loss_fn,device):
    model.eval()
    total_loss=0
    total_correct=0
    total_samples=0
    
    with torch.no_grad():
        for image,label in test_loader:
            image=image.to(device)
            label=label.to(device)
            logit,embedding=model(image)
            loss=loss_fn(logit,label)
            
            total_loss+=loss.item()*image.size(0)
            prediction=torch.argmax(logit,dim=1)
            total_correct+=(prediction==label).sum().item()
            total_samples+=image.size(0)

    avg_loss=total_loss/total_samples
    accuracy=total_correct/total_samples
    return avg_loss,accuracy

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    train_loader,train_dataset=Dataloader(
        data_dir="data/train",
        image_size=64,
        batch_size=128,
        shuffle=True
    )
    test_loader,test_dataset=Dataloader(
        data_dir="data/test",
        image_size=64,
        batch_size=128,
        shuffle=False
    )
    model=Detector(num_class=2)
    model=model.to(device)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    best_accuracy=0
    best_epoch=0
    epochs=5

    for epoch in range(epochs):
        train_loss,train_accuracy=train(model,train_loader,loss_fn,optimizer,device)
        test_loss,test_accuracy=test(model,test_loader,loss_fn,device)
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"train loss={train_loss:.4f} "
            f"train accuracy={train_accuracy:.4f} "
            f"test loss={test_loss:.4f} "
            f"test accuracy={test_accuracy:.4f} "
        )
        if test_accuracy>best_accuracy:
            best_accuracy=test_accuracy
            best_epoch=epoch
            torch.save(model.state_dict(),"detector.pth")

    print(f"best test accuracy={best_accuracy:.4f} emerged in epoch {best_epoch+1}")

if __name__=="__main__":
    main()