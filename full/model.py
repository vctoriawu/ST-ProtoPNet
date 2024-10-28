import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, \
    resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, \
    densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, \
    vgg16_bn_features, \
    vgg19_features, vgg19_bn_features

from util.receptive_field import compute_proto_layer_rf_info_v2
from util import lorentz as L

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}


class STProtoPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 num_global_prototypes_per_class: int = 1,
                 learn_curv: bool = True,
                 curv_init : float = 1.0):

        super(STProtoPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function  # log

        assert (self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features  #

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers_trivial = nn.Sequential(
                nn.Identity() if 'VGG' in features_name else nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
            )
            self.add_on_layers_support = nn.Sequential(
                nn.Identity() if 'VGG' in features_name else nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
            )

        self.prototype_vectors_trivial = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.prototype_vectors_support = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.last_layer_trivial = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        self.last_layer_support = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # dimension of the global prototypes is the same as the dimension of the part prototypes but the number of global prototypes is different
        self.num_global_attn_map = num_global_prototypes_per_class
        self.global_prototype_shape = (num_global_prototypes_per_class * self.num_classes,  # PG, for now G=1.
                                       self.prototype_shape[1], 1, 1)
        
        # shape  (PG=num_classes*G, D, 1, 1)
        self.global_prototype_vectors_trivial = nn.Parameter(torch.rand(self.global_prototype_shape), requires_grad=True)
        self.global_prototype_vectors_support = nn.Parameter(torch.rand(self.global_prototype_shape), requires_grad=True)

        self.global_attn_module_trivial = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1] // 2,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1] // 2,
                out_channels=self.num_global_attn_map,
                kernel_size=1,
                bias=False,
            ),
        )

        self.global_attn_module_support = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1] // 2,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1] // 2,
                out_channels=self.num_global_attn_map,
                kernel_size=1,
                bias=False,
            ),
        )

        if init_weights:
            self._initialize_weights()

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        # Learnable scalars to ensure that image features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(self.prototype_shape[1] ** -0.5).log())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def conv_features(self, x):

        x = self.features(x)
        x_trivial = self.add_on_layers_trivial(x)     
        x_support = self.add_on_layers_support(x)

        return x_trivial, x_support
    
    def get_global_attn_map(self, part_conv_features_trivial, part_conv_features_support):
        # shape (N, G, H, W)   G=num_global_prots
        global_attn_map_trivial = self.global_attn_module_trivial(part_conv_features_trivial)
        global_attn_map_support = self.global_attn_module_support(part_conv_features_support)

        # TODO experiment if absolute value is neeeded
        global_attn_map_trivial = torch.abs(global_attn_map_trivial).unsqueeze(2)  # shape (N, G, 1, H, W)
        global_attn_map_support = torch.abs(global_attn_map_support).unsqueeze(2)
        return global_attn_map_trivial, global_attn_map_support

    def _cosine_convolution(self, prototypes, x):

        x = F.normalize(x, p=2, dim=1)
        prototype_vectors = F.normalize(prototypes, p=2, dim=1)
        similarity = F.conv2d(input=x, weight=prototype_vectors)

        return similarity

    def prototype_distances(self, x):

        ##### MERU-based operations for hyperbolic space analysis #####
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)

        part_conv_features_trivial, part_conv_features_support = self.conv_features(x)

         ### global-level features
        global_attn_map_triv, global_attn_map_supp = self.get_global_attn_map(part_conv_features_trivial, part_conv_features_support)  # shape (N, G, 1, H, W)
        global_triv_feats_extracted = (global_attn_map_triv * part_conv_features_trivial.unsqueeze(1)).sum(dim=3).sum(dim=3)  # shape (N, G, D)
        global_supp_feats_extracted = (global_attn_map_supp * part_conv_features_support.unsqueeze(1)).sum(dim=3).sum(dim=3)  # shape (N, G, D)

        # Lift the conv features onto the hyperbloid
        hyperbolic_part_trivial_feats = L.get_hyperbolic_feats(part_conv_features_trivial.permute(0,2,3,1), self.visual_alpha, self.curv, self.device)
        hyperbolic_part_support_feats = L.get_hyperbolic_feats(part_conv_features_support.permute(0,2,3,1), self.visual_alpha, self.curv, self.device)

        # Lift the global features onto the hyperbloid
        hyperbolic_global_trivial_feats = L.get_hyperbolic_feats(global_triv_feats_extracted, self.visual_alpha, self.curv, self.device)
        hyperbolic_global_support_feats = L.get_hyperbolic_feats(global_supp_feats_extracted, self.visual_alpha, self.curv, self.device)

        # Calculate the lorentz distance instead of cosine
        # cosine_similarities_trivial = self._cosine_convolution(self.prototype_vectors_trivial, conv_features_trivial)
        # cosine_similarities_support = self._cosine_convolution(self.prototype_vectors_support, conv_features_support)
        hyper_prototypes_triv = L.get_hyperbolic_feats(self.prototype_vectors_trivial.squeeze(),self.visual_alpha, self.curv, self.device)
                
        part_feat_prot_triv_lorentz_distance = L.pairwise_dist(hyperbolic_part_trivial_feats,
                                                               hyper_prototypes_triv,
                                                              #self.prototype_vectors_trivial.squeeze(),
                                                              _curv)
        # global min pooling (spatial across HxW)
        part_feat_prot_triv_lorentz_distance = part_feat_prot_triv_lorentz_distance.permute(0,3,1,2)  # Shape (N, P, H, W)
        # N, P, H, W = part_feat_prot_triv_lorentz_distance.shape
        # part_triv_min_lorentz_distances = -F.max_pool2d(-part_feat_prot_triv_lorentz_distance, kernel_size=(H, W))  # shape (N, P, 1, 1)
        # part_triv_min_lorentz_distances = part_triv_min_lorentz_distances.view(-1, self.num_prototypes)  # shape (N, P)
        # part_triv_prototype_activations = self.distance_2_similarity(part_triv_min_lorentz_distances)  # shape (N, P)
        hyper_prototypes_supp = L.get_hyperbolic_feats(self.prototype_vectors_support.squeeze(),self.visual_alpha, self.curv, self.device)        
        part_feat_prot_supp_lorentz_distance = L.pairwise_dist(hyperbolic_part_support_feats,
                                                               hyper_prototypes_supp,
                                                              #self.prototype_vectors_support.squeeze(),
                                                              _curv)
        # global min pooling (spatial across HxW)
        part_feat_prot_supp_lorentz_distance = part_feat_prot_supp_lorentz_distance.permute(0,3,1,2)  # Shape (N, P, H, W)
        # N, P, H, W = part_feat_prot_supp_lorentz_distance.shape
        # part_supp_min_lorentz_distances = -F.max_pool2d(-part_feat_prot_supp_lorentz_distance, kernel_size=(H, W))  # shape (N, P, 1, 1)
        # part_supp_min_lorentz_distances = part_supp_min_lorentz_distances.view(-1, self.num_prototypes)  # shape (N, P)
        # part_supp_prototype_activations = self.distance_2_similarity(part_supp_min_lorentz_distances)  # shape (N, P)
        
        # Relu from Deformable ProtoPNet: https://github.com/jdonnelly36/Deformable-ProtoPNet/blob/main/model.py
        ################################################
        lorentz_trivial = torch.relu(part_feat_prot_triv_lorentz_distance)
        lorentz_support = torch.relu(part_feat_prot_supp_lorentz_distance)
        ################################################

        return lorentz_trivial, lorentz_support

    def distance_2_similarity(self, distances):

        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise Exception('other activation function NOT implemented')

    def distance_2_similarity_linear(self, distances):
        return (self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]) ** 2 - distances

    def global_min_pooling(self, input):

        min_output = -F.max_pool2d(-input, kernel_size=(input.size()[2], input.size()[3]))
        min_output = min_output.view(-1, self.num_prototypes)

        return min_output

    def global_max_pooling(self, input):

        max_output = F.max_pool2d(input, kernel_size=(input.size()[2], input.size()[3]))
        max_output = max_output.view(-1, self.num_prototypes)

        return max_output

    def forward(self, x):

        lorentz_trivial, lorentz_support = self.prototype_distances(x)

        prototype_activations_trivial = self.global_max_pooling(lorentz_trivial)
        prototype_activations_support = self.global_max_pooling(lorentz_support)

        logits_trivial = self.last_layer_trivial(prototype_activations_trivial)
        logits_support = self.last_layer_support(prototype_activations_support)

        return (logits_trivial, logits_support), (prototype_activations_trivial, prototype_activations_support), \
               (lorentz_trivial, lorentz_support)

    def push_forward_trivial(self, x):

        conv_features_trivial, _ = self.conv_features(x)  # [batchsize, 64, 14, 14]

        similarities = self._cosine_convolution(self.prototype_vectors_trivial, conv_features_trivial)
        distances = - similarities

        conv_output = F.normalize(conv_features_trivial, p=2, dim=1)

        return conv_output, distances

    def push_forward_support(self, x):

        _, conv_features_support = self.conv_features(x)  # [batchsize, 64, 14, 14]

        similarities = self._cosine_convolution(self.prototype_vectors_support, conv_features_support)
        distances = - similarities

        conv_output = F.normalize(conv_features_support, p=2, dim=1)

        return conv_output, distances


    def set_last_layer_incorrect_connection(self, incorrect_strength):

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer_trivial.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
        self.last_layer_support.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers_trivial.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.add_on_layers_support.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)   # 0.0


def construct_STProtoPNet(base_architecture, pretrained=True, img_size=224,
                          prototype_shape=(2000, 128, 1, 1), num_classes=200,
                          prototype_activation_function='log',
                          add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return STProtoPNet(features=features,
                       img_size=img_size,
                       prototype_shape=prototype_shape,
                       proto_layer_rf_info=proto_layer_rf_info,
                       num_classes=num_classes,
                       init_weights=True,
                       prototype_activation_function=prototype_activation_function,
                       add_on_layers_type=add_on_layers_type)

