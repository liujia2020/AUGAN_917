import torch
from .base_model import BaseModel
from . import network
import torchvision
from thop import profile


class Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', 
                            netG='unet_128', 
                            netD='basic', 
                            use_sab=False, 
                            name='unet_b002')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla') # 不使用缓冲池功能。vanilla是最原始的GAN论文中提出的损失函数，使用二元交叉熵（Binary Cross-Entropy）
            parser.add_argument('--lambda_L1', type=float, default=1, help='weight for L1 loss')
            # L1 损失用于惩罚生成图像与真实图像之间的像素级差异，使得生成器生成的图像在细节上更接近真实图像。
            # 添加一个名为 lambda_L1 的参数，用于控制 L1 损失在总损失中的权重，默认为1。
        return parser

    def __init__(self, opt): # opt是一个对象，是一个集中的配置管理器。
        BaseModel.__init__(self, opt) # 调用父类的初始化方法
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.feature_extractor = network.FeatureExtractor(vgg)
        
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        self.netG = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.use_sab)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = network.define_D(opt.input_nc + opt.output_nc, 
                                         opt.ndf, 
                                         opt.netD,
                                         opt.n_layers_D, 
                                         opt.norm, 
                                         opt.init_type, 
                                         opt.init_gain, 
                                         self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()  # (L1损失): 计算生成图像和真实图像之间每个像素差异的绝对值之和。
            self.criterionL2 = torch.nn.MSELoss() # 计算生成图像和真实图像之间每个像素差异的平方和。
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, data, target):
        self.input_image = data.to(self.device)
        self.target_image = target.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.input_image)  # G(A)
        # self.total_ops, self.total_params = profile(self.netG.cuda(), (self.input_image.cuda(),))
        # print(self.total_ops, self.total_params)



    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.input_image, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach()) # 拼接后的图像，传入判别器，得到预测值  .detach() 用于防止在更新判别器时计算生成器的梯度。
        self.loss_D_fake = self.criterionGAN(pred_fake, False) # 计算判别器在假图像的损失。
        
        # Real
        input_imageB = torch.cat((self.input_image, self.target_image), 1)
        pred_real = self.netD(input_imageB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) *0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.input_image, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.target_image) * self.opt.lambda_L1

        # vgg  = torchvision.models.vgg19(pretrained=True)
        # vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        #
        # feature_extractor = network.FeatureExtractor(vgg)
        self.target_imagef = self.feature_extractor(self.target_image.cpu()).cuda()
        self.fake_Bf = self.feature_extractor(self.fake_B.cpu()).cuda()
        pred_fake1 = torch.cat((self.target_imagef,self.fake_Bf),1)
        self.contentLoss = self.criterionGAN(pred_fake1,True)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.contentLoss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward() # compute fake image G(A)
        
        # update D
        self.set_requires_grad(self.netD, True) # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradient to zero
        self.backward_D()   # calculate gradient for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G

        self.optimizer_G.step()             # udpate G's weights#