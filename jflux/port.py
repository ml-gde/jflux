from einops import rearrange


def port_attn_block(attn_block, tensors, prefix):
    # port the norm
    attn_block.norm.scale.value = tensors[f"{prefix}.norm.weight"]
    attn_block.norm.bias.value = tensors[f"{prefix}.norm.bias"]

    # port the k, q, v layers
    attn_block.k.kernel.value = rearrange(
        tensors[f"{prefix}.k.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    attn_block.k.bias.value = tensors[f"{prefix}.k.bias"]

    attn_block.q.kernel.value = rearrange(
        tensors[f"{prefix}.q.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    attn_block.q.bias.value = tensors[f"{prefix}.q.weight"]

    attn_block.v.kernel.value = rearrange(
        tensors[f"{prefix}.v.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    attn_block.v.bias.value = tensors[f"{prefix}.v.weight"]

    # port the proj_out layer
    attn_block.proj_out.kernel.value = rearrange(
        tensors[f"{prefix}.proj_out.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    attn_block.proj_out.bias.value = tensors[f"{prefix}.proj_out.weight"]

    return attn_block


def port_resent_block(resnet_block, tensors, prefix):
    # port the norm
    resnet_block.norm1.scale.value = tensors[f"{prefix}.norm1.weight"]
    resnet_block.norm1.bias.value = tensors[f"{prefix}.norm1.bias"]

    resnet_block.norm2.scale.value = tensors[f"{prefix}.norm2.weight"]
    resnet_block.norm2.bias.value = tensors[f"{prefix}.norm2.bias"]

    # port the convs
    resnet_block.conv1.kernel.value = rearrange(
        tensors[f"{prefix}.conv1.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    resnet_block.conv1.bias.value = tensors[f"{prefix}.conv1.weight"]

    resnet_block.conv2.kernel.value = rearrange(
        tensors[f"{prefix}.conv2.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    resnet_block.conv2.bias.value = tensors[f"{prefix}.conv2.weight"]

    if resnet_block.in_channels != resnet_block.out_channels:
        resnet_block.nin_shortcut.kernel.value = rearrange(
            tensors[f"{prefix}.nin_shortcut.weight"], "i o k1 k2 -> k1 k2 o i"
        )
        resnet_block.nin_shortcut.bias.value = tensors[f"{prefix}.nin_shortcut.bias"]

    return resnet_block


def port_downsample(downsample, tensors, prefix):
    # port the conv
    downsample.conv.kernel.value = rearrange(
        tensors[f"{prefix}.conv.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    downsample.conv.bias.value = tensors[f"{prefix}.conv.bias"]
    return downsample


def port_upsample(upsample, tensors, prefix):
    # port the conv
    upsample.conv.kernel.value = rearrange(
        tensors[f"{prefix}.conv.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    upsample.conv.bias.value = tensors[f"{prefix}.conv.bias"]
    return upsample


def port_encoder(encoder, tensors, prefix):
    # port downsampling
    conv_in = encoder.conv_in
    conv_in.kernel.value = rearrange(
        tensors[f"{prefix}.conv_in.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    conv_in.bias.value = tensors[f"{prefix}.conv_in.bias"]

    # down
    down = encoder.down
    for i in range(len(down.layers)):
        # block
        block = down.layers[i].block
        for j in range(len(block.layers)):
            resnet_block = block.layers[j]
            resnet_block = port_resent_block(
                resnet_block=resnet_block,
                tensors=tensors,
                prefix=f"{prefix}.down.{i}.block.{j}",
            )

        # attn
        attn = down.layers[i].attn
        for j in range(len(attn.layers)):
            attn_block = attn.layers[j]
            attn_block = port_attn_block(
                attn_block=attn_block,
                tensors=tensors,
                prefix=f"{prefix}.attn.{i}.block.{j}",
            )

        # downsample
        if i != encoder.num_resolutions - 1:
            downsample = down.layers[i].downsample
            downsample = port_downsample(
                downsample=downsample,
                tensors=tensors,
                prefix=f"{prefix}.down.{i}.downsample",
            )

    # mid
    mid = encoder.mid
    mid_block_1 = mid.block_1
    mid_block_1 = port_resent_block(
        resnet_block=mid_block_1, tensors=tensors, prefix=f"{prefix}.mid.block_1"
    )

    mid_attn_1 = mid.attn_1
    mid_attn_1 = port_attn_block(
        attn_block=mid_attn_1, tensors=tensors, prefix=f"{prefix}.mid.attn_1"
    )

    mid_block_2 = mid.block_2
    mid_block_2 = port_resent_block(
        resnet_block=mid_block_2, tensors=tensors, prefix=f"{prefix}.mid.block_2"
    )

    # norm out
    norm_out = encoder.norm_out
    norm_out.scale.value = tensors[f"{prefix}.norm_out.weight"]
    norm_out.bias.value = tensors[f"{prefix}.norm_out.bias"]

    # conv out
    conv_out = encoder.conv_out
    conv_out.kernel.value = rearrange(
        tensors[f"{prefix}.conv_out.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    conv_out.bias.value = tensors[f"{prefix}.conv_out.bias"]

    return encoder


def port_decoder(decoder, tensors, prefix):
    # port downsampling
    conv_in = decoder.conv_in

    conv_in.kernel.value = rearrange(
        tensors[f"{prefix}.conv_in.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    conv_in.bias.value = tensors[f"{prefix}.conv_in.bias"]

    # mid
    mid = decoder.mid

    mid_block_1 = mid.block_1
    mid_block_1 = port_resent_block(
        resnet_block=mid_block_1, tensors=tensors, prefix=f"{prefix}.mid.block_1"
    )

    mid_attn_1 = mid.attn_1
    mid_attn_1 = port_attn_block(
        attn_block=mid_attn_1, tensors=tensors, prefix=f"{prefix}.mid.attn_1"
    )

    mid_block_2 = mid.block_2
    mid_block_2 = port_resent_block(
        resnet_block=mid_block_2, tensors=tensors, prefix=f"{prefix}.mid.block_2"
    )

    # up
    up = decoder.up

    for i in range(len(up.layers)):
        # block
        block = up.layers[i].block
        for j in range(len(block.layers)):
            resnet_block = block.layers[j]
            resnet_block = port_resent_block(
                resnet_block=resnet_block,
                tensors=tensors,
                prefix=f"{prefix}.up.{i}.block.{j}",
            )

        # attn
        attn = up.layers[i].attn
        for j in range(len(attn.layers)):
            attn_block = attn.layers[j]
            attn_block = port_attn_block(
                attn_block=attn_block,
                tensors=tensors,
                prefix=f"{prefix}.up.{i}.attn.{j}",
            )

        # upsample
        if i != 0:
            upsample = up.layers[i].upsample
            upsample = port_upsample(
                upsample=upsample, tensors=tensors, prefix=f"{prefix}.up.{i}.upsample"
            )

    # norm out
    norm_out = decoder.norm_out
    norm_out.scale.value = tensors[f"{prefix}.norm_out.weight"]
    norm_out.bias.value = tensors[f"{prefix}.norm_out.bias"]

    # conv out
    conv_out = decoder.conv_out
    conv_out.kernel.value = rearrange(
        tensors[f"{prefix}.conv_out.weight"], "i o k1 k2 -> k1 k2 o i"
    )
    conv_out.bias.value = tensors[f"{prefix}.conv_out.bias"]

    return decoder


def port_autoencoder(autoencoder, tensors):
    autoencoder.encoder = port_encoder(
        encoder=autoencoder.encoder, tensors=tensors, prefix="encoder"
    )
    autoencoder.decoder = port_decoder(
        decoder=autoencoder.decoder, tensors=tensors, prefix="decoder"
    )
    return autoencoder
