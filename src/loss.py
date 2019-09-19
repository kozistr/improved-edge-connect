import torch
import torch.nn as nn
import torchvision.models as models


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, loss_type: str = 'nsgan', target_real_label: float = 1., target_fake_label: float = .0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = loss_type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if self.type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif self.type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif self.type == 'hinge':
            self.criterion = nn.ReLU()
        else:
            raise NotImplementedError

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    @staticmethod
    def compute_gram(x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_t = f.transpose(1, 2)
        gram = f.bmm(f_t) / (h * w * ch)
        return gram

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = .0
        style_loss += self.criterion(self.compute_gram(x_vgg['conv2_2']), self.compute_gram(y_vgg['conv2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['conv3_4']), self.compute_gram(y_vgg['conv3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['conv4_4']), self.compute_gram(y_vgg['conv4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['conv5_2']), self.compute_gram(y_vgg['conv5_2']))
        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf
    """
    def __init__(self, weights=(1., 1., 1., 1., 1.)):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = .0
        content_loss += self.weights[0] * self.criterion(x_vgg['conv1_1'], y_vgg['conv1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['conv2_1'], y_vgg['conv2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['conv3_1'], y_vgg['conv3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['conv4_1'], y_vgg['conv4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['conv5_1'], y_vgg['conv5_1'])
        return content_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.conv1_1 = torch.nn.Sequential()
        self.conv1_2 = torch.nn.Sequential()

        self.conv2_1 = torch.nn.Sequential()
        self.conv2_2 = torch.nn.Sequential()

        self.conv3_1 = torch.nn.Sequential()
        self.conv3_2 = torch.nn.Sequential()
        self.conv3_3 = torch.nn.Sequential()
        self.conv3_4 = torch.nn.Sequential()

        self.conv4_1 = torch.nn.Sequential()
        self.conv4_2 = torch.nn.Sequential()
        self.conv4_3 = torch.nn.Sequential()
        self.conv4_4 = torch.nn.Sequential()

        self.conv5_1 = torch.nn.Sequential()
        self.conv5_2 = torch.nn.Sequential()
        self.conv5_3 = torch.nn.Sequential()
        self.conv5_4 = torch.nn.Sequential()

        for x in range(1):
            self.conv1_1.add_module(str(x), features[x])

        for x in range(1, 3):
            self.conv1_2.add_module(str(x), features[x])

        for x in range(3, 6):
            self.conv2_1.add_module(str(x), features[x])

        for x in range(6, 8):
            self.conv2_2.add_module(str(x), features[x])

        for x in range(8, 11):
            self.conv3_1.add_module(str(x), features[x])

        for x in range(11, 13):
            self.conv3_2.add_module(str(x), features[x])

        for x in range(13, 15):
            self.conv3_3.add_module(str(x), features[x])

        for x in range(15, 17):
            self.conv3_4.add_module(str(x), features[x])

        for x in range(17, 20):
            self.conv4_1.add_module(str(x), features[x])

        for x in range(20, 22):
            self.conv4_2.add_module(str(x), features[x])

        for x in range(22, 24):
            self.conv4_3.add_module(str(x), features[x])

        for x in range(24, 26):
            self.conv4_4.add_module(str(x), features[x])

        for x in range(26, 29):
            self.conv5_1.add_module(str(x), features[x])

        for x in range(29, 31):
            self.conv5_2.add_module(str(x), features[x])

        for x in range(31, 33):
            self.conv5_3.add_module(str(x), features[x])

        for x in range(33, 35):
            self.conv5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)

        conv2_1 = self.conv2_1(conv1_2)
        conv2_2 = self.conv2_2(conv2_1)

        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)

        conv4_1 = self.conv4_1(conv3_4)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        conv4_4 = self.conv4_4(conv4_3)

        conv5_1 = self.conv5_1(conv4_4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        conv5_4 = self.conv5_4(conv5_3)

        out = {
            'conv1_1': conv1_1,
            'conv1_2': conv1_2,

            'conv2_1': conv2_1,
            'conv2_2': conv2_2,

            'conv3_1': conv3_1,
            'conv3_2': conv3_2,
            'conv3_3': conv3_3,
            'conv3_4': conv3_4,

            'conv4_1': conv4_1,
            'conv4_2': conv4_2,
            'conv4_3': conv4_3,
            'conv4_4': conv4_4,

            'conv5_1': conv5_1,
            'conv5_2': conv5_2,
            'conv5_3': conv5_3,
            'conv5_4': conv5_4,
        }
        return out
