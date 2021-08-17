

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNet_rfs(nn.Module):

	def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
	             dropblock_size=5, num_classes=-1, use_se=False):
		super(ResNet_rfs, self).__init__()

		self.inplanes = 3
		self.use_se = use_se
		self.layer1 = self._make_layer(block, n_blocks[0], 64,
		                               stride=2, drop_rate=drop_rate)
		self.layer2 = self._make_layer(block, n_blocks[1], 160,
		                               stride=2, drop_rate=drop_rate)
		self.layer3 = self._make_layer(block, n_blocks[2], 320,
		                               stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
		self.layer4 = self._make_layer(block, n_blocks[3], 640,
		                               stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
		if avg_pool:
			# self.avgpool = nn.AvgPool2d(5, stride=1)
			self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.keep_prob = keep_prob
		self.keep_avg_pool = avg_pool
		self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
		self.drop_rate = drop_rate

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.num_classes = num_classes
		if self.num_classes > 0:
			self.classifier = nn.Linear(640, self.num_classes)

	def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		if n_block == 1:
			layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
		else:
			layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
		layers.append(layer)
		self.inplanes = planes * block.expansion

		for i in range(1, n_block):
			if i == n_block - 1:
				layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
				              block_size=block_size, use_se=self.use_se)
			else:
				layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
			layers.append(layer)

		return nn.Sequential(*layers)

	def forward(self, x, is_feat=False):
		x = self.layer1(x)
		f0 = x
		x = self.layer2(x)
		f1 = x
		x = self.layer3(x)
		f2 = x
		x = self.layer4(x)
		f3 = x
		if self.keep_avg_pool:
			x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		feat = x
		return feat
		# if self.num_classes > 0:
		# 	x = self.classifier(x)
		#
		# if is_feat:
		# 	return [f0, f1, f2, f3, feat], x
		# else:
		# 	return x
