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
    def __init__(self,D ,N ,head , args=None, logger=None):
        super(MHA_,self).__init__()
        
        self.D = int(D)
        self.Dh = int(D / head)
        self.head = int(head)
        self.N = int(N)
        self.softmax = nn.Softmax(0)
        self.scale = self.Dh ** 0.5
        # z is N*D                        or (N+1)*D
        # W matrix is D*Dh                , Dh = D/head
        # Q,K,V = zW = (N*D)*(D*Dh) = N*Dh
        # A = ScoreMatrix = (N*Dh)*(Dh*N) = N*N
        # AV = (N*N)*(N*Dh) = N*Dh
        # concate head => output is N*D

        self.Wq_1   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_2   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_3   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_4   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_5   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_6   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_7   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_8   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_9   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_10  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_11  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wq_12  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)

        self.Wk_1   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_2   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_3   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_4   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_5   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_6   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_7   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_8   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_9   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_10  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_11  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wk_12  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)

        self.Wv_1   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_2   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_3   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_4   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_5   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_6   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_7   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_8   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_9   = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_10  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_11  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Wv_12  = Conv2d(self.Dh,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)

        self.QK_T_1  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_2  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_3  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_4  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_5  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_6  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_7  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_8  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_9  = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_10 = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_11 = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.QK_T_12 = Conv2d(self.Dh,N,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)

        self.Sc_V_1  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_2  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_3  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_4  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_5  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_6  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_7  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_8  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_9  = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_10 = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_11 = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.Sc_V_12 = Conv2d(N,self.Dh,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
    def forward(self, x):
        # x is 1*N*D
        # For pytorch, D*(N*1)   cause pytorch is channel first

        x = x.permute(2,1,0).reshape(1,self.D,self.N,1)
        
        x_ = torch.chunk(x,self.head,dim=1)
        # x_= 1*Dh*N*1
        # For pytorch, Q,K,V = 1*Dh*(N*1)
        q_1 = self.Wq_1(x_[0])
        k_1 = self.Wk_1(x_[0])
        v_1 = self.Wv_1(x_[0])
        q_2 = self.Wq_2(x_[1])
        k_2 = self.Wk_2(x_[1])
        v_2 = self.Wv_2(x_[1])
        q_3 = self.Wq_3(x_[2])
        k_3 = self.Wk_3(x_[2])
        v_3 = self.Wv_3(x_[2])
        q_4 = self.Wq_4(x_[3])
        k_4 = self.Wk_4(x_[3])
        v_4 = self.Wv_4(x_[3])
        q_5 = self.Wq_5(x_[4])
        k_5 = self.Wk_5(x_[4])
        v_5 = self.Wv_5(x_[4])
        q_6 = self.Wq_6(x_[5])
        k_6 = self.Wk_6(x_[5])
        v_6 = self.Wv_6(x_[5])
        q_7 = self.Wq_7(x_[6])
        k_7 = self.Wk_7(x_[6])
        v_7 = self.Wv_7(x_[6])
        q_8 = self.Wq_8(x_[7])
        k_8 = self.Wk_8(x_[7])
        v_8 = self.Wv_8(x_[7])
        q_9 = self.Wq_9(x_[8])
        k_9 = self.Wk_9(x_[8])
        v_9 = self.Wv_9(x_[8])
        q_10 = self.Wq_10(x_[9])
        k_10 = self.Wk_10(x_[9])
        v_10 = self.Wv_10(x_[9])
        q_11 = self.Wq_11(x_[10])
        k_11 = self.Wk_11(x_[10])
        v_11 = self.Wv_11(x_[10])
        q_12 = self.Wq_12(x_[11])
        k_12 = self.Wk_12(x_[11])
        v_12 = self.Wv_12(x_[11])
        # MMM : self.F(x, weight) weight is  (N,Dh,1,1)=(out_ch,in_ch,kH,kW)
        # need to be N*C*H*W => N*Dh*1*1
        k_1 = k_1.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_2 = k_2.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_3 = k_3.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_4 = k_4.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_5 = k_5.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_6 = k_6.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_7 = k_7.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_8 = k_8.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_9 = k_9.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_10 = k_10.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_11 = k_11.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        k_12 = k_12.permute(2,1,0,3) # .reshape(self.N_2,self.Dh,1,1)
        Sc_1 = self.QK_T_1(q_1,nn.Parameter(k_1,requires_grad=False))
        Sc_2 = self.QK_T_2(q_2,nn.Parameter(k_2,requires_grad=False)) 
        Sc_3 = self.QK_T_3(q_3,nn.Parameter(k_3,requires_grad=False))
        Sc_4 = self.QK_T_4(q_4,nn.Parameter(k_4,requires_grad=False)) 
        Sc_5 = self.QK_T_5(q_5,nn.Parameter(k_5,requires_grad=False))
        Sc_6 = self.QK_T_6(q_6,nn.Parameter(k_6,requires_grad=False))
        Sc_7 = self.QK_T_7(q_7,nn.Parameter(k_7,requires_grad=False))
        Sc_8 = self.QK_T_8(q_8,nn.Parameter(k_8,requires_grad=False)) 
        Sc_9 = self.QK_T_9(q_9,nn.Parameter(k_9,requires_grad=False))
        Sc_10 = self.QK_T_10(q_10,nn.Parameter(k_10,requires_grad=False)) 
        Sc_11 = self.QK_T_11(q_11,nn.Parameter(k_11,requires_grad=False))
        Sc_12 = self.QK_T_12(q_12,nn.Parameter(k_12,requires_grad=False)) 
        # (  Q     *   K_T)   *     V
        # (N_1*Dh) * (Dh*N_2) * (N_3*Dh)
        # Sc dim is 1*N_2*N_1*1, so N_2 dim needs to be apply softmax
        # N and N need swap axis
        Sc_1  = Sc_1.squeeze()
        Sc_2  = Sc_2.squeeze()
        Sc_3  = Sc_3.squeeze()
        Sc_4  = Sc_4.squeeze()
        Sc_5  = Sc_5.squeeze()
        Sc_6  = Sc_6.squeeze()
        Sc_7  = Sc_7.squeeze()
        Sc_8  = Sc_8.squeeze()
        Sc_9  = Sc_9.squeeze()
        Sc_10 = Sc_10.squeeze()
        Sc_11 = Sc_11.squeeze()
        Sc_12 = Sc_12.squeeze()
        Sc_1  = self.softmax(Sc_1  / self.scale).reshape(self.N,self.N,1,1)
        Sc_2  = self.softmax(Sc_2  / self.scale).reshape(self.N,self.N,1,1)
        Sc_3  = self.softmax(Sc_3  / self.scale).reshape(self.N,self.N,1,1)
        Sc_4  = self.softmax(Sc_4  / self.scale).reshape(self.N,self.N,1,1)
        Sc_5  = self.softmax(Sc_5  / self.scale).reshape(self.N,self.N,1,1)
        Sc_6  = self.softmax(Sc_6  / self.scale).reshape(self.N,self.N,1,1)
        Sc_7  = self.softmax(Sc_7  / self.scale).reshape(self.N,self.N,1,1)
        Sc_8  = self.softmax(Sc_8  / self.scale).reshape(self.N,self.N,1,1)
        Sc_9  = self.softmax(Sc_9  / self.scale).reshape(self.N,self.N,1,1)
        Sc_10 = self.softmax(Sc_10 / self.scale).reshape(self.N,self.N,1,1)
        Sc_11 = self.softmax(Sc_11 / self.scale).reshape(self.N,self.N,1,1)
        Sc_12 = self.softmax(Sc_12 / self.scale).reshape(self.N,self.N,1,1)
        # v_ permute to 1*N*Dh*1
        v_1  = v_1.permute(0,2,1,3)
        v_2  = v_2.permute(0,2,1,3)
        v_3  = v_3.permute(0,2,1,3)
        v_4  = v_4.permute(0,2,1,3)
        v_5  = v_5.permute(0,2,1,3)
        v_6  = v_6.permute(0,2,1,3)
        v_7  = v_7.permute(0,2,1,3)
        v_8  = v_8.permute(0,2,1,3)
        v_9  = v_9.permute(0,2,1,3)
        v_10 = v_10.permute(0,2,1,3)
        v_11 = v_11.permute(0,2,1,3)
        v_12 = v_12.permute(0,2,1,3)
        # out_ is 1*N*Dh*1
        out_1  = self.Sc_V_1(v_1,nn.Parameter(Sc_1,requires_grad=False))
        out_2  = self.Sc_V_2(v_2,nn.Parameter(Sc_2,requires_grad=False))
        out_3  = self.Sc_V_3(v_3,nn.Parameter(Sc_3,requires_grad=False))
        out_4  = self.Sc_V_4(v_4,nn.Parameter(Sc_4,requires_grad=False))
        out_5  = self.Sc_V_5(v_5,nn.Parameter(Sc_5,requires_grad=False))
        out_6  = self.Sc_V_6(v_6,nn.Parameter(Sc_6,requires_grad=False))
        out_7  = self.Sc_V_7(v_7,nn.Parameter(Sc_7,requires_grad=False))
        out_8  = self.Sc_V_8(v_8,nn.Parameter(Sc_8,requires_grad=False))
        out_9  = self.Sc_V_9(v_9,nn.Parameter(Sc_9,requires_grad=False))
        out_10 = self.Sc_V_10(v_10,nn.Parameter(Sc_10,requires_grad=False))
        out_11 = self.Sc_V_11(v_11,nn.Parameter(Sc_11,requires_grad=False))
        out_12 = self.Sc_V_12(v_12,nn.Parameter(Sc_12,requires_grad=False))

        out = torch.concat((out_1,out_2,out_3,out_4,out_5,out_6,out_7,out_8,out_9,out_10,out_11,out_12),axis = 2).reshape(1,self.N,self.D)
        return out




class ViT_one_layer(nn.Module):
    def __init__(self, args, num_classes,N,logger,D,MLP_hidden,head):
        super(ViT_one_layer, self).__init__()
        self.num_classes = num_classes
        self.MLP_hidden = int(MLP_hidden)
        self.D = int(D)
        self.N = N
        self.head = head
        self.ln1_1 = nn.LayerNorm(self.D)
        self.ln1_2 = nn.LayerNorm(self.D)
        self.MHA = self.make_attention_layer_(MHA_,logger=logger,args=args)
        self.fc1  = Conv2d(self.D, self.MLP_hidden, kernel_size=(1,1),stride=(1,1),padding=0,args=args,logger=logger)
        self.GELU = nn.GELU()
        self.fc2  = Conv2d(self.MLP_hidden, self.D, kernel_size=(1,1),stride=(1,1),padding=0,args=args,logger=logger)
    def make_attention_layer_(self,block,logger=None,args=None):
        layers = []
        layers.append(block(self.D,self.N,head=self.head,args=args,logger=logger))

        return nn.Sequential(*layers)

    def forward(self,x):
        # x dim is 1*N*D
        identify_1 = self.ln1_1(x)
        x = self.MHA(identify_1) + identify_1
        identify_2 = self.ln1_2(x)
        # x dim is 1*N*D, need to permute to 1*D*(N*1)
        identify_2 = identify_2.permute(0,2,1).reshape(1,self.D,self.N,1)
        x = self.fc1(identify_2)
        # x dim is 1*MLP_hidden*(N*1)
        x = self.fc2(self.GELU(x))
        # x dim is 1*D*(N*1)
        out = x + identify_2
        out = out.permute(0,2,1,3).reshape(1,self.N,self.D)
        return out # 1*N*D



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


class ViT(nn.Module):
    def __init__(self, args, chw, n_patches, n_block, num_classes, MLP_hidden, logger,head,D):
        super(ViT, self).__init__()
        self.D = int(D)
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
        self.linear_mapper = Conv2d(int(self.input_d), int(self.D),kernel_size=(1,1),stride=(1,1),padding=0,args=args,logger=logger)
        self.class_token = nn.Parameter(torch.rand(1, self.D))

        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches **2 + 1, self.D)))
        self.pos_embed.requires_grad = False
        
        self.blocks = nn.ModuleList([ViT_one_layer(args=args, logger=logger, N=self.n_patches**2+1, head=self.head, MLP_hidden=self.MLP_hidden,D=self.D,num_classes=self.num_classes) for _ in range(self.n_block)])
        self.classifier = Conv2d(self.D, self.num_classes,args=args,logger=logger,kernel_size=(1,1),stride=(1,1),padding=0)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, images):
        n, c, h ,w = images.shape
        patches = patchify(images, self.n_patches).cuda()

        patches = patches.permute(2,1,0)
        temp_size = patches.shape
        patches = patches.reshape(1,temp_size[0],temp_size[1],temp_size[2])
        tokens = self.linear_mapper(patches) 
        temp_size = tokens.shape
        tokens = tokens.reshape(temp_size[1],temp_size[2],temp_size[3])
        tokens = tokens.permute(2,1,0)

        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range (len(tokens))])

        pos_embed = self.pos_embed.repeat(n,1,1) 
        out = tokens + pos_embed

        for block in self.blocks:
            out = block(out)
        # out is 1*N*D
        out_shape = out.shape
        out = out.permute(0,2,1).reshape(1,out_shape[2],out_shape[1],1)
        out = self.classifier(out).squeeze()
        out = out[:, 0].reshape(1,self.num_classes)
        return self.softmax(out)



