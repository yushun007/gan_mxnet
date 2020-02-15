import mxnet as mx
import random
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import math
from mxnet import ndarray as nd
from mxnet import gluon, random, image
from mxnet.gluon import nn, loss as gloss, Trainer as T
from mxnet.gluon import data as gdata
from mxnet.gluon.data.vision import transforms as TF
from mxnet import autograd, initializer, optimizer

manualSeed = 999
random.seed(manualSeed)
mx.random.seed(manualSeed)

dataroot = './data/'
workers = 16
batch_size = 64
image_size = 64
nchannels = 3
nz = 100
ngf = 64
ndf = 64
num_epoch = 5
lr = 0.0002
beta1 = 0.5
ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
transforms = TF.Compose([TF.Resize(image_size), TF.CenterCrop(image_size), TF.ToTensor(), TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
# ,TF.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))


irange = range


def is_ndarray(obj):
    return isinstance(obj, nd.ndarray.NDArray)


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    if not (is_ndarray(tensor) or (isinstance(tensor, list) and all(is_ndarray(t) for t in tensor))):

        raise TypeError(
            'tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = nd.stack(tensor, dim=0)

    if tensor.ndim == 2:  # single image H x W
        tensor = nd.expand_dims(tensor, axis=0)
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = nd.concat(tensor, tensor, tensor, dim=0)
        tensor = nd.expand_dims(tensor, axis=0)

    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = nd.concat(tensor, tensor, tensor, dim=1)

    if normalize is True:
        tensor = tensor.copy()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(
                range, tuple),                 "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            nd.clip(img, min, max)
            img += (-min)
            img /= (max-min + 1e-5)
            #img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min().asscalar()),
                        float(t.max().asscalar()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.shape[0] == 1:
        return tensor.reshape((-3, -2))

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]  # 我们截取的mini_batch大小
    # print(nmaps)
    xmaps = min(nrow, nmaps)  # 输入的参数
    ymaps = int(math.ceil(float(nmaps) / xmaps))  # 算列数向下取整
    height, width = int(
        tensor.shape[2] + padding), int(tensor.shape[3] + padding)  # 图片显示的高宽
    num_channels = tensor.shape[1]  # 图像通道数
    grid = nd.full((num_channels, height * ymaps + padding, width * xmaps +
                    padding), pad_value)  # 创建一个全为零的通道数等于输入图片，高宽等于图片高宽分别乘以行列数加上图片的间隔
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid[:, x * width + padding: x * width + padding + width - padding, y *
                 height + padding: y * height + padding+height - padding] = tensor[k]
            k = k + 1
    return grid


dataset = gdata.vision.datasets.ImageFolderDataset(root=dataroot)
dataloader = gdata.DataLoader(dataset.transform_first(
    transforms), batch_size=batch_size, shuffle=True, num_workers=workers)
# datalodaer 返回一个mini_batch的迭代器，使用python的iter迭代器接口读取
realbatch = next(iter(dataloader))
# image_ = image.imread('./data/img_align_celeba/168835.jpg')
# print(type(image_),image_)
# print(type(image.imdecode(image_.asnumpy())),image.imdecode(image_.asnumpy()))
# print(type(image_.asnumpy()),image_.asnumpy())
# plt.imshow(image_.asnumpy())
# plt.show()


plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Traning Images")
grid_ = make_grid(realbatch[0][:64], padding=2, normalize=True)
print(grid_.context)
# image_ = np.transpose( grid_.asnumpy(),(1,2,0))
# plt.imshow(image_)
print(realbatch[0][:64].shape)
plt.imshow(np.transpose(make_grid(
    realbatch[0][:64], padding=2, normalize=True).as_in_context(mx.cpu()).asnumpy(), (1, 2, 0)))
plt.show()


class Generator(nn.Block):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential()
        self.main.add(nn.Conv2DTranspose(ngf*8, 4, 1, 0, use_bias=False),
                      nn.BatchNorm(),
                      nn.Activation('relu'),
                      nn.Conv2DTranspose(ngf*4, 4, 2, 1, use_bias=False),
                      nn.BatchNorm(),
                      nn.Activation('relu'),
                      nn.Conv2DTranspose(ngf*2, 4, 2, 1, use_bias=False),
                      nn.BatchNorm(),
                      nn.Activation('relu'),
                      nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False),
                      nn.BatchNorm(),
                      nn.Activation('relu'),
                      nn.Conv2DTranspose(nchannels, 4, 2, 1, use_bias=False, activation='tanh'))

    def forward(self, input):
        return self.main(input)


netG = Generator()
netG.initialize(init=initializer.Normal(0.02), ctx=ctx)
# print(netG)


class Discriminator(nn.Block):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential()
        self.main.add(
            nn.Conv2D(ndf, 4, 2, 1, use_bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2D(ndf*2, 4, 2, 1, use_bias=False),
            nn.BatchNorm(),
            nn.LeakyReLU(0.2),
            nn.Conv2D(ndf*4, 4, 2, 1, use_bias=False),
            nn.BatchNorm(),
            nn.LeakyReLU(0.2),
            nn.Conv2D(ndf*8, 4, 2, 1, use_bias=False),
            nn.BatchNorm(),
            nn.LeakyReLU(0.2),
            nn.Conv2D(1, 4, 1, 0, use_bias=False, activation='sigmoid')
        )

    def forward(self, input):
        return self.main(input)


netD = Discriminator()
netD.initialize(init=initializer.Normal(0.002), ctx=ctx)
# print(netD)

criteion = gloss.SigmoidBinaryCrossEntropyLoss()

fixed_noise = nd.random_normal(0, 1, (64, nz, 1, 1), ctx=ctx)
# print(fixed_noise.shape)


tranerG = gluon.Trainer(netG.collect_params(), 'adam', {
                        'learning_rate': lr, 'beta1': beta1})
tranerD = gluon.Trainer(netD.collect_params(), 'adam', {
                        'learning_rate': lr, 'beta1': beta1})

img_list = []
G_losses = []
D_losses = []
iters = 0
print("Starting Traning Loop ...")


for epoch in range(num_epoch):
    for i, data in enumerate(dataloader, 0):
        real_cpu = data[0].as_in_context(ctx)
        # print("real_cpu:",real_cpu.context)

        b_size = real_cpu.shape[0]
        real_label = nd.ones(b_size, ctx=ctx)
        fake_label = nd.zeros(b_size, ctx=ctx)
        # print(real_label.shape,fake_label.shape)
        noise = nd.random_normal(0, 1, (b_size, nz, 1, 1), ctx=ctx)
        errG = nd.random_normal(0, 1, (b_size, nz, 1, 1), ctx=ctx)
        errD = nd.random_normal(0, 1, (b_size, nz, 1, 1), ctx=ctx)
        D_G_z1 = 0
        D_G_z2 = 0
        D_x = 0
        if i % 3 != 0:

            with autograd.record():
                output = netD(real_cpu).reshape(-1, 1)
                # print('output.shape',netD(real_cpu).shape,'real_label.shape',real_label.shape)
                errD_real = criteion(output, real_label)

                D_x = output.mean().asscalar()
                fake = netG(noise)
                output = netD(fake.detach()).reshape(-1, 1)
                errD_fake = criteion(output, fake_label)
                errD = errD_fake + errD_real
                errD.backward()
            # print("errD.shape",errD.shape)
            tranerD.step(b_size)
            D_G_z1 = output.mean().asscalar()

        else:
            with autograd.record():
                # print('noise.shape',noise.shape)
                z = netG(noise)
                output = netD(z).reshape(-1, 1)
                # print('output.shape',output.shape)
                errG = criteion(output, real_label)
                errG.backward()
            D_G_z2 = output.mean().asscalar()
            tranerG.step(b_size)

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)):%.4f/%.4f' % (epoch,
                                                                                                 num_epoch, i, len(dataloader), errD.mean().asscalar(), errG.mean().asscalar(), D_x, D_G_z1, D_G_z2))
        G_losses.append(errG.mean().asscalar())
        D_losses.append(errD.mean().asscalar())

        if (iters % 500 == 0) or ((epoch == num_epoch - 1) and (i == len(dataloader)-1)):

            fake = netG(fixed_noise).detach().as_in_context(mx.cpu())
            img_list.append(make_grid(fake, padding=2,
                                      normalize=True).asnumpy())

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
       for i in img_list]
ani = animation.ArtistAnimation(
    fig, ims, interval=1000, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())


real_batch = next(iter(dataloader))
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Image")
plt.imshow(np.transpose(
    make_grid(real_batch[0][:64], padding=5, normalize=True).asnumpy(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
