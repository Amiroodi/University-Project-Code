import torch
import torchvision
from torch import nn

def reg_classify(x, device):
    bins = torch.tensor([0.5, 1.5, 2.5, 3.5]).to(device)  # Class boundaries
    # Classify using bucketize
    classified = torch.bucketize(x, bins, right=False)  # right=False ensures correct bin placement
    return classified

class ThreeHeadCNN(nn.Module):
    def __init__(self, device):
        super(ThreeHeadCNN, self).__init__()

        self.device = device

        # Load EfficientNet encoder
        weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
        efficientNet = torchvision.models.efficientnet_b1(weights=weights)
        self.encoder = efficientNet.features

        # Pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.batch_norm_1= nn.BatchNorm1d(1280) 
        self.batch_norm_2= nn.BatchNorm1d(1280)

        self.dense1 = nn.Linear(1280 * 2, 512)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 5) # 5 output nodes for classification
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1) # 1 output node is for regression
        )

        # Ordinal regression head
        self.ordinal_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 5) # 5 output nodes for ordinal regression
        )

        # Final regression head
        self.final_head = nn.Sequential(
            nn.Linear(11, 1)
        )

    def forward(self, x):
        x = self.encoder(x) # Extract features

        # Apply pooling layers
        max_pooled = self.global_max_pool(x).view(x.size(0), -1)
        avg_pooled = self.global_avg_pool(x).view(x.size(0), -1)

        # Concatenate
        x1 = self.batch_norm_1(max_pooled)
        x2 = self.batch_norm_2(avg_pooled)
        x = torch.concat([x1, x2], dim=1)
        x = torch.relu(self.dense1(x))

        # enc_out for visualizing data with t-SNE
        enc_out = x

        # Classification branch
        class_out = self.classification_head(x)

        # Regression branch
        reg_out = self.regression_head(x).squeeze(dim=1) # Single value

        # Ordinal regression branch
        ord_out = self.ordinal_head(x)

        # Final regression head branch
        y_pred_class = (torch.argmax(torch.softmax(class_out, dim=1), dim=1))
        y_pred_reg = (reg_classify(reg_out, device=self.device).to(self.device))
        y_pred_ord = (torch.sum(torch.round(torch.sigmoid(ord_out)), dim=1, keepdim=True).squeeze(dim=1) - 1)

        # y_pred_class = torch.softmax(class_out, dim=1)
        # y_pred_reg = reg_classify(reg_out, device=self.device).to(self.device)
        # y_pred_ord = torch.sigmoid(ord_out)

        # print('y_pred_class is :', y_pred_class)
        # print('y_pred_reg is :', y_pred_reg)
        # print('y_pred_ord is :', y_pred_ord)

        x = torch.stack((y_pred_class, y_pred_reg, y_pred_ord), dim=1)

        final_out = self.final_head(x).squeeze(dim=1)

        # print('x is: ', x)
        # print('final_out is: ', final_out)

        return class_out, reg_out, ord_out, enc_out, final_out

    