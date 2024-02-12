import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from utils.Parser import parse_args
from utils.checkpoints import load_checkpoint, save_checkpoint
from model.Unet import UNet

from dataset.PetDataset import OxfordIIITPetsAugmented
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_args()

pets_path_train = os.path.join('OxfordPets', 'train')
pets_path_test = os.path.join('OxfordPets', 'test')
#divide dataset i train and test
pets_train_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_train, split="trainval", 
                                                     target_types="segmentation", download=False)
pets_test_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", target_types="segmentation", 
                                                    download=False)

# Create a tensor for a segmentation trimap.
# Input: Float tensor with values in [0.0 .. 1.0]
# Output: Long tensor with values in {0, 1, 2}
#0 = Pet pixel (segmentation target)
#1 = background pixel
#2 = border (ambiguous) region pixel
def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def args_to_dict(**kwargs):
    return kwargs

transform_dict = args_to_dict(
    pre_transform=T.ToTensor(), #transform to tensor
    pre_target_transform=T.ToTensor(),
    common_transform=T.Compose([
        T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
        # Random Horizontal Flip as data augmentation.
        T.RandomHorizontalFlip(p=0.5),
    ]),
    post_transform=T.Compose([
        # Color Jitter as data augmentation.
        T.ColorJitter(contrast=0.3), #change constrast
    ]),
    post_target_transform=T.Compose([
        T.Lambda(tensor_trimap),
    ]),
)

pets_train = OxfordIIITPetsAugmented(
    root=pets_path_train,
    split="trainval",
    target_types="segmentation",
    download=False,
    **transform_dict,
)
pets_test = OxfordIIITPetsAugmented(
    root=pets_path_test,
    split="test",
    target_types="segmentation",
    download=False,
    **transform_dict,
)

pets_train_loader = DataLoader(
    pets_train,
    batch_size=64,
    shuffle=True,
)
pets_test_loader = DataLoader(
    pets_test,
    batch_size=21,
    shuffle=True,
)

#Custom IoU metric that is differentiable so that it can be used as a loss functions
#Use the probability of the predicted class to determine the degree of overlap
def IoUMetric(pred, gt, softmax=False):
    # Run softmax if input is logits.
    if softmax is True:
        pred = nn.Softmax(dim=1)(pred)
    # end if
    
    # Add the one-hot encoded masks for all 3 output channels
    # (for all the classes) to a tensor named 'gt' (ground truth).
    gt = torch.cat([ (gt == i) for i in range(3) ], dim=1)
    # print(f"[2] Pred shape: {pred.shape}, gt shape: {gt.shape}")

    intersection = gt * pred
    union = gt + pred - intersection

    # Compute the sum over all the dimensions except for the batch dimension.
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)
    
    # Compute the mean over the batch dimension.
    return iou.mean()

class IoULoss(nn.Module):
    def __init__(self, softmax=False):
        super().__init__()
        self.softmax = softmax
    
    # pred => Predictions (logits, B, 3, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        # return 1.0 - IoUMetric(pred, gt, self.softmax)
        # Compute the negative log loss for stable training.
        return -(IoUMetric(pred, gt, self.softmax).log())
    # end def
# end class

#Define Model, optimizer and criterion   
model = UNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
criterion = IoULoss(softmax=True)

#FOlder where the logs will be saved
runs_folder = "./res/" + time.strftime("%Y%m%d-%H%M")

if not os.path.exists(runs_folder):
    os.makedirs(runs_folder)
#TensorBoard
writer = SummaryWriter(runs_folder)

# Start training
print('Start training')
print('Parameters: epochs= {}, bs= {}, lr= {}'.format(args.epochs, args.b, args.lr))

if args.resume:
    load_checkpoint(model, args)

train_loss = []
train_acc = []
val_loss = []
val_acc = []

# train loop
for epoch in range(args.epochs):
    # losses and accuracy for each epoch
    trainloss = 0
    valloss = 0
    trainaccuracy = 0
    valaccuracy = 0

    model.train()

    loop = tqdm(enumerate(pets_train_loader, 0), total = len(pets_train_loader), leave = False)

    for i, (images, targets) in loop:

        images = images.to(device)
        targets = targets.to(device) # the ground truth mask

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        trainloss += loss.item()

        metric = IoUMetric(outputs, targets)
        trainaccuracy += metric.item()
        
        writer.add_scalar('Loss/train', trainloss / (i+1), epoch * len(pets_train_loader) + i)
        writer.add_scalar('Accuracy/train', trainaccuracy / (i+1), epoch * len(pets_train_loader) + i)

        if i % 200 == 199:
            print("[it: {}] loss: {} accuracy: {}".format(i+1, trainloss / (i+1), trainaccuracy / (i+1)))
    
    print("Epoch : {} finished train, starting eval".format(epoch))
    # Save checkpoint every 5 epochs
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        save_checkpoint(model, args.save_weights, epoch)
    train_loss.append(trainloss/len(pets_train_loader))
    train_acc.append(trainaccuracy/len(pets_train_loader))

    model.eval()
    loop = tqdm(enumerate(pets_test_loader, 0), total = len(pets_train_loader), leave = False)

    with torch.no_grad():
        for i, (images, targets) in loop:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            
            loss = criterion(outputs, targets)
            valloss += loss.item()

            metric = IoUMetric(outputs, targets)
            valaccuracy += metric.item()

            writer.add_scalar('Loss/val', valloss / (i+1), epoch * len(pets_test_loader) + i)
            writer.add_scalar('Accuracy/val', valaccuracy / (i+1), epoch * len(pets_test_loader) + i)

            if i % 100 == 99:
                print("[it: {}] loss: {} accuracy: {}".format(i+1, valloss / (i+1), valaccuracy / (i+1)))

        val_loss.append(valloss/len(pets_test_loader))
        val_acc.append(valaccuracy/len(pets_test_loader))

    print("Epoch : {} , train loss : {} , valid loss : {} , train accuracy: {} , val accuracy: {}".format(epoch, train_loss[-1], val_loss[-1], train_acc[-1], val_acc[-1]))

writer.close()
print("Finished training")

