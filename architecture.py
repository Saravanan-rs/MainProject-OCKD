import torch.nn as nn
class StudentNet(nn.Module):
    def __init__(self, ic=3):   
        super(StudentNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ic, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),           
        )
        
        self.Block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def embedding(self, x):
        x0 = self.conv1(x)
        x1 = self.Block1(x0)	    	    	
        x2 = self.Block2(x1)	    
        x3 = self.Block3(x2)
        
        return x1, x2, x3
    
    def resize(self, x1, x2, x3):
        x1 = self.downsample(x1)
        x1 = self.downsample(x1)
        x2 = self.downsample(x2)
        x3 = x3
        
        return x1, x2, x3
    
    def forward(self, x):
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	
        x_Block1_f = self.downsample(x_Block1)
        x_Block1_f = self.downsample(x_Block1_f)
        
        x_Block2 = self.Block2(x_Block1)	   
        x_Block2_f = self.downsample(x_Block2)   
        
        x_Block3 = self.Block3(x_Block2)
        x_Block3_f = x_Block3 
        
        return x_Block1_f, x_Block2_f, x_Block3_f
