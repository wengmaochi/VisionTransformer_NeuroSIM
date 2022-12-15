from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch
import numpy as np
name = 0 
def conv1x1(in_planes, out_planes, stride=1,args=None,logger=None, weight_matrix=None):
    """1x1 convolution"""
    global name
    if args.mode == "WAGE":        
        conv2d = QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                         wl_error=args.wl_error,wl_weight= weight_matrix,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                         name = 'Conv1x1'+'_'+str(name)+'_', model = args.model)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                         logger=logger,wl_input = args.wl_activate,wl_weight= weight_matrix,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                         name = 'Conv1x1'+'_'+str(name)+'_' )
    name += 1
    return conv2d

def Conv2d(in_planes, out_planes, kernel_size, stride, padding, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":        
        conv2d = QConv2d(in_planes, out_planes, kernel_size, stride, padding, logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                         wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                         name = 'Conv'+'_'+str(name)+'_', model = args.model)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False,
                         logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                         name = 'Conv'+'_'+str(name)+'_' )
    name += 1
    return conv2d

def Linear(in_planes, out_planes, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":
        linear = QLinear(in_planes, out_planes, 
                        logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, 
                        name='FC'+'_'+str(name)+'_', model = args.model)
    elif args.mode == "FP":
        linear = FLinear(in_planes, out_planes, bias=False,
                        logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                        name='FC'+'_'+str(name)+'_')
    name += 1
    return linear
    






class MHA_(nn.Module):
    def __init__(self,D ,N , args=None, logger=None,head=12):
        super(MHA_,self).__init__()
        
        self.D = D
        self.Dh = int(D / head)
        self.head = head
        self.N = N

        # z is N*D                        or (N+1)*D
        # W matrix is D*Dh                , Dh = D/head
        # Q,K,V = zW = (N*D)*(D*Dh) = N*Dh
        # A = ScoreMatrix = (N*Dh)*(Dh*N) = N*N
        # AV = (N*N)*(N*Dh) = N*Dh
        # concate head => output is N*D

        self.Wq_1   = Linear(self.Dh,self.Dh,args=args,logger=logger)
        self.Wq_2   = Linear(self.Dh,self.Dh,args=args,logger=logger)
        self.Wk_1   = Linear(self.Dh,self.Dh,args=args,logger=logger)
        self.Wk_2   = Linear(self.Dh,self.Dh,args=args,logger=logger)
        self.Wv_1   = Linear(self.Dh,self.Dh,args=args,logger=logger)
        self.Wv_2   = Linear(self.Dh,self.Dh,args=args,logger=logger)

        self.QK_T_1 = Linear(self.Dh,N,args=args,logger=logger)
        self.QK_T_2 = Linear(self.Dh,N,args=args,logger=logger)
        
        self.Sc_V_1 = Linear(N,self.Dh,args=args,logger=logger)
        self.Sc_V_2 = Linear(N,self.Dh,args=args,logger=logger)
    def forward(self, x):
        # x is N*D
        x_1,x_2 = torch.chunk(x,self.head,dim=2)
        # x_1 is N*Dh
        # For pytorch, Q,K,V = N*Dh
        q_1 = self.Wq_1(x_1)
        k_1 = self.Wk_1(x_1)
        v_1 = self.Wv_1(x_1)

        q_2 = self.Wq_2(x_2)
        k_2 = self.Wk_2(x_2)
        v_2 = self.Wv_2(x_2)
        
        # MMM : self.F(x, weight) weight is (out_feature, in_feature)  =>  N*Dh
        # need to be N*C*H*W
        k_1 = k_1.reshape(k_1.shape[1],k_1.shape[2])
        k_2 = k_2.reshape(k_2.shape[1],k_2.shape[2])
        Sc_1 = self.QK_T_1(q_1,nn.Parameter(k_1,requires_grad=False))
        Sc_2 = self.QK_T_2(q_2,nn.Parameter(k_2,requires_grad=False)) 
        # Sc dim is N*N*1
        # N and N need swap axis
        # Sc_1 = Sc_1.permute(1,0,2).reshape(self.N,self.N,1,1)
        # Sc_2 = Sc_2.permute(1,0,2).reshape(self.N,self.N,1,1)
        # out_ is N*Dh*1
        Sc_1 = Sc_1.reshape(Sc_1.shape[1],Sc_1.shape[2])
        Sc_2 = Sc_2.reshape(Sc_2.shape[1],Sc_2.shape[2])
        out_1 = self.Sc_V_1(v_1.transpose(-2,-1),nn.Parameter(Sc_1,requires_grad=False))
        out_2 = self.Sc_V_2(v_2.transpose(-2,-1),nn.Parameter(Sc_2,requires_grad=False))
        out = torch.concat((out_1,out_2),axis = 1)
        out = out.transpose(-2,-1)
        out = out.squeeze()
        return out




class ViT_one_layer(nn.Module):
    def __init__(self, args, logger,N,head,MLP_hidden=3072,D=768):
        super(ViT_one_layer, self).__init__()
        self.ln1_1 = nn.LayerNorm(D)
        self.ln1_2 = nn.LayerNorm(D)
        self.MHA = self.make_attention_layer_(MHA_,logger=logger,args=args,N=N,head=head,D=D)
        self.fc1  = Linear(D,MLP_hidden,args=args,logger=logger)
        self.GELU  = nn.GELU()
        self.fc2  = Linear(MLP_hidden,D,args=args, logger=logger)



    def make_attention_layer_(self,block,N,logger=None,args=None,head=None,D=None):

        layers = []
        layers.append(block(D=D,N=N,args=args,logger=logger,head=head))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.ln1_1(x)
        identify_1 = x
        x = self.MHA(x)
        identify_2 = x + identify_1
        x = self.ln1_2(identify_2)
        x = self.fc1(x)
        x = self.GELU(x)
        x = self.fc2(x)
        out = x + identify_2
        return out


def get_positional_embeddings(sequence_length,d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1)/d)))
    return result

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h==w, "Patchfy method is implemented for square images only"

    patches = torch.zeros(n, n_patches**2 , c* h*w // n_patches **2)
    patch_size = h // n_patches
    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i*patch_size : (i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches
    
    # for i in range(n_patches):
    #     for j in range(n_patches):
    #         patch = images[:, i*patch_size : (i+1)*patch_size, j*patch_size:(j+1)*patch_size]
    #         patches[]
    # return patch


class ViT(nn.Module):
    def __init__(self, args, chw, n_patches, n_block, num_classes, MLP_hidden, logger,N,head,D=768):
        super(ViT, self).__init__()
        self.D = D
        self.head = head
        self.chw = chw
        self.n_patches = n_patches
        self.n_block = n_block
        self.num_classes = int(num_classes)
        self.MLP_hidden = MLP_hidden

        assert chw[1] % n_patches == 0, "input shape is not divisible by number of patches"
        assert chw[2] % n_patches == 0, "input shape is not divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        
        self.input_d = int(chw[0]* self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = Linear(self.input_d, self.D,args=args, logger=logger)
        # self.linear_mapper = Conv2d(int(self.input_d), int(self.D),kernel_size=(1,1),stride=(1,1),padding=0,args=args,logger=logger)
        self.class_token = nn.Parameter(torch.rand(1, self.D))

        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches **2 + 1, self.D)))
        self.pos_embed.requires_grad = False
        
        self.blocks = nn.ModuleList([ViT_one_layer(args=args, logger=logger, N=self.n_patches**2+1, head=self.head, MLP_hidden=self.MLP_hidden,D=self.D)])
        self.classifier = Linear(self.D, self.num_classes,args=args,logger=logger)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, images):
        n, c, h ,w = images.shape
        patches = patchify(images, self.n_patches).cuda()
        
        # patches = patches.permute(2,1,0)
        temp_size = patches.shape
        # patches = patches.reshape(1,temp_size[0],temp_size[1],temp_size[2])
        tokens = self.linear_mapper(patches) 
        # print(tokens.shape)
        # temp_size = tokens.shape
        # tokens = tokens.reshape(temp_size[1],temp_size[2],temp_size[3])
        # tokens = tokens.permute(2,1,0)
        # print(tokens.shape)
        # print("---------------------ha")

        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range (len(tokens))])

        pos_embed = self.pos_embed.repeat(n,1,1) 
        out = tokens + pos_embed

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]

        out = self.classifier(out)
        return self.softmax(out)



