layer_info = [
    ['block1_conv1', '(224, 224, 64)', '1792'],
    ['block1_conv2', '(224, 224, 64)', '36928'],
    ['block1_pool', '(112, 112, 64)', '0'],
    ['block2_conv1', '(112, 112, 128)', '73856'],
    ['block2_conv2', '(112, 112, 128)', '147584'],
    ['block2_pool', '(56, 56, 128)', '0'],
    ['block3_conv1', '(56, 56, 256)', '295168'],
    ['block3_conv2', '(56, 56, 256)', '590080'],
    ['block3_conv3', '(56, 56, 256)', '590080'],
    ['block3_pool', '(28, 28, 256)', '0'],
    ['block4_conv1', '(28, 28, 512)', '1180160'],
    ['block4_conv2', '(28, 28, 512)', '2359808'],
    ['block4_conv3', '(28, 28, 512)', '2359808'],
    ['block4_pool', '(28, 28, 256)', '0'],
    ['block5_conv1', '(14, 14, 512)', '2359808'],
    ['block5_conv2', '(14, 14, 512)', '2359808'],
    ['block5_conv3', '(14, 14, 512)', '2359808'],
    ['block5_pool', '(7, 7, 512)', '0'],
    ['flatten', '(25088)', '0'],
    ['fc_1', '(4096)', '0'],
    ['fc_2', '(4096)', '0'],
    ['predictions', '(1000)', '0']
];


def prepare_history(hist):
    layer_index = int(hist['params']['layer_index']) - 1
    cur_layer_info = layer_info[layer_index]

    hist['layer_info'] = {}
    hist['layer_info']['name'] = cur_layer_info[0]
    hist['layer_info']['shape'] = cur_layer_info[1]
    hist['layer_info']['param_count'] = cur_layer_info[2]

    return hist


def prepare_histories(histories, all=False):
    histories = list(reversed([prepare_history(h) for h in histories]))
    if len(histories) > 4 and not all:
        return histories[:4]
    else:
        return histories