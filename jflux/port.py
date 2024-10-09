from einops import rearrange


##############################################################################################
# AUTOENCODER MODEL PORTING
##############################################################################################


def port_group_norm(group_norm, tensors, prefix):
    group_norm.scale.value = tensors[f"{prefix}.weight"]
    group_norm.bias.value = tensors[f"{prefix}.bias"]

    return group_norm


def port_conv(conv, tensors, prefix):
    conv.kernel.value = rearrange(tensors[f"{prefix}.weight"], "i o k1 k2 -> k1 k2 o i")
    conv.bias.value = tensors[f"{prefix}.bias"]

    return conv


def port_attn_block(attn_block, tensors, prefix):
    # port the norm
    attn_block.norm = port_group_norm(
        group_norm=attn_block.norm,
        tensors=tensors,
        prefix=f"{prefix}.norm",
    )

    # port the k, q, v layers
    attn_block.k = port_conv(
        conv=attn_block.k,
        tensors=tensors,
        prefix=f"{prefix}.k",
    )

    attn_block.q = port_conv(
        conv=attn_block.q,
        tensors=tensors,
        prefix=f"{prefix}.q",
    )

    attn_block.v = port_conv(
        conv=attn_block.v,
        tensors=tensors,
        prefix=f"{prefix}.v",
    )

    # port the proj_out layer
    attn_block.proj_out = port_conv(
        conv=attn_block.proj_out,
        tensors=tensors,
        prefix=f"{prefix}.proj_out",
    )

    return attn_block


def port_resent_block(resnet_block, tensors, prefix):
    # port the norm
    resnet_block.norm1 = port_group_norm(
        group_norm=resnet_block.norm1,
        tensors=tensors,
        prefix=f"{prefix}.norm1",
    )
    resnet_block.norm2 = port_group_norm(
        group_norm=resnet_block.norm2,
        tensors=tensors,
        prefix=f"{prefix}.norm2",
    )

    # port the convs
    resnet_block.conv1 = port_conv(
        conv=resnet_block.conv1,
        tensors=tensors,
        prefix=f"{prefix}.conv1",
    )
    resnet_block.conv2 = port_conv(
        conv=resnet_block.conv2,
        tensors=tensors,
        prefix=f"{prefix}.conv2",
    )

    if resnet_block.in_channels != resnet_block.out_channels:
        resnet_block.nin_shortcut = port_conv(
            conv=resnet_block.nin_shortcut,
            tensors=tensors,
            prefix=f"{prefix}.nin_shortcut",
        )

    return resnet_block


def port_downsample(downsample, tensors, prefix):
    # port the conv
    downsample.conv = port_conv(
        conv=downsample.conv,
        tensors=tensors,
        prefix=f"{prefix}.conv",
    )

    return downsample


def port_upsample(upsample, tensors, prefix):
    # port the conv
    upsample.conv = port_conv(
        conv=upsample.conv,
        tensors=tensors,
        prefix=f"{prefix}.conv",
    )

    return upsample


def port_encoder(encoder, tensors, prefix):
    # conv in
    encoder.conv_in = port_conv(
        conv=encoder.conv_in,
        tensors=tensors,
        prefix=f"{prefix}.conv_in",
    )

    # down
    for i, down_layer in enumerate(encoder.down.layers):
        # block
        for j, block_layer in enumerate(down_layer.block.layers):
            block_layer = port_resent_block(
                resnet_block=block_layer,
                tensors=tensors,
                prefix=f"{prefix}.down.{i}.block.{j}",
            )
        # attn
        for j, attn_layer in enumerate(down_layer.attn.layers):
            attn_layer = port_attn_block(
                attn_block=attn_layer,
                tensors=tensors,
                prefix=f"{prefix}.attn.{i}.block.{j}",
            )

        # downsample
        if i != encoder.num_resolutions - 1:
            downsample = down_layer.downsample
            downsample = port_downsample(
                downsample=downsample,
                tensors=tensors,
                prefix=f"{prefix}.down.{i}.downsample",
            )

    # mid
    encoder.mid.block_1 = port_resent_block(
        resnet_block=encoder.mid.block_1,
        tensors=tensors,
        prefix=f"{prefix}.mid.block_1",
    )
    encoder.mid.attn_1 = port_attn_block(
        attn_block=encoder.mid.attn_1,
        tensors=tensors,
        prefix=f"{prefix}.mid.attn_1",
    )
    encoder.mid.block_2 = port_resent_block(
        resnet_block=encoder.mid.block_2,
        tensors=tensors,
        prefix=f"{prefix}.mid.block_2",
    )

    # norm out
    encoder.norm_out = port_group_norm(
        group_norm=encoder.norm_out,
        tensors=tensors,
        prefix=f"{prefix}.norm_out",
    )

    # conv out
    encoder.conv_out = port_conv(
        conv=encoder.conv_out,
        tensors=tensors,
        prefix=f"{prefix}.conv_out",
    )

    return encoder


def port_decoder(decoder, tensors, prefix):
    # conv in
    decoder.conv_in = port_conv(
        conv=decoder.conv_in,
        tensors=tensors,
        prefix=f"{prefix}.conv_in",
    )

    # mid
    decoder.mid.block_1 = port_resent_block(
        resnet_block=decoder.mid.block_1,
        tensors=tensors,
        prefix=f"{prefix}.mid.block_1",
    )
    decoder.mid.attn_1 = port_attn_block(
        attn_block=decoder.mid.attn_1,
        tensors=tensors,
        prefix=f"{prefix}.mid.attn_1",
    )
    decoder.mid.block_2 = port_resent_block(
        resnet_block=decoder.mid.block_2,
        tensors=tensors,
        prefix=f"{prefix}.mid.block_2",
    )

    for i, up_layer in enumerate(decoder.up.layers):
        # block
        for j, block_layer in enumerate(up_layer.block.layers):
            block_layer = port_resent_block(
                resnet_block=block_layer,
                tensors=tensors,
                prefix=f"{prefix}.up.{i}.block.{j}",
            )

        # attn
        for j, attn_layer in enumerate(up_layer.attn.layers):
            attn_layer = port_attn_block(
                attn_block=attn_layer,
                tensors=tensors,
                prefix=f"{prefix}.up.{i}.attn.{j}",
            )

        # upsample
        if i != 0:
            up_layer.upsample = port_upsample(
                upsample=up_layer.upsample,
                tensors=tensors,
                prefix=f"{prefix}.up.{i}.upsample",
            )

    # norm out
    decoder.norm_out = port_group_norm(
        group_norm=decoder.norm_out,
        tensors=tensors,
        prefix=f"{prefix}.norm_out",
    )

    # conv out
    decoder.conv_out = port_conv(
        conv=decoder.conv_out,
        tensors=tensors,
        prefix=f"{prefix}.conv_out",
    )

    return decoder


def port_autoencoder(autoencoder, tensors):
    autoencoder.encoder = port_encoder(
        encoder=autoencoder.encoder,
        tensors=tensors,
        prefix="encoder",
    )
    autoencoder.decoder = port_decoder(
        decoder=autoencoder.decoder,
        tensors=tensors,
        prefix="decoder",
    )
    return autoencoder


##############################################################################################
# FLUX MODEL PORTING
##############################################################################################


def port_linear(linear, tensors, prefix):
    linear.kernel.value = rearrange(tensors[f"{prefix}.weight"], "i o -> o i")
    linear.bias.value = tensors[f"{prefix}.bias"]
    return linear


def port_modulation(modulation, tensors, prefix):
    modulation.lin = port_linear(
        linear=modulation.lin, tensors=tensors, prefix=f"{prefix}.lin"
    )
    return modulation


def port_rms_norm(rms_norm, tensors, prefix):
    rms_norm.scale.value = tensors[f"{prefix}.scale"]
    return rms_norm


def port_qk_norm(qk_norm, tensors, prefix):
    qk_norm.query_norm = port_rms_norm(
        rms_norm=qk_norm.query_norm,
        tensors=tensors,
        prefix=f"{prefix}.query_norm",
    )
    qk_norm.key_norm = port_rms_norm(
        rms_norm=qk_norm.key_norm,
        tensors=tensors,
        prefix=f"{prefix}.key_norm",
    )
    return qk_norm


def port_self_attention(self_attention, tensors, prefix):
    self_attention.qkv = port_linear(
        linear=self_attention.qkv,
        tensors=tensors,
        prefix=f"{prefix}.qkv",
    )

    self_attention.norm = port_qk_norm(
        qk_norm=self_attention.norm,
        tensors=tensors,
        prefix=f"{prefix}.norm",
    )

    self_attention.proj = port_linear(
        linear=self_attention.proj,
        tensors=tensors,
        prefix=f"{prefix}.proj",
    )

    return self_attention


def port_double_stream_block(double_stream_block, tensors, prefix):
    double_stream_block.img_mod = port_modulation(
        modulation=double_stream_block.img_mod,
        tensors=tensors,
        prefix=f"{prefix}.img_mod",
    )

    # double_stream_block.img_norm1 has no params

    double_stream_block.img_attn = port_self_attention(
        self_attention=double_stream_block.img_attn,
        tensors=tensors,
        prefix=f"{prefix}.img_attn",
    )

    # double_stream_block.img_norm2 has no params

    double_stream_block.img_mlp.layers[0] = port_linear(
        linear=double_stream_block.img_mlp.layers[0],
        tensors=tensors,
        prefix=f"{prefix}.img_mlp.0",
    )
    double_stream_block.img_mlp.layers[2] = port_linear(
        linear=double_stream_block.img_mlp.layers[2],
        tensors=tensors,
        prefix=f"{prefix}.img_mlp.2",
    )

    double_stream_block.txt_mod = port_modulation(
        modulation=double_stream_block.txt_mod,
        tensors=tensors,
        prefix=f"{prefix}.txt_mod",
    )

    # double_stream_block.txt_norm1 has no params

    double_stream_block.txt_attn = port_self_attention(
        self_attention=double_stream_block.txt_attn,
        tensors=tensors,
        prefix=f"{prefix}.txt_attn",
    )

    # double_stream_block.txt_norm2 has no params

    double_stream_block.txt_mlp.layers[0] = port_linear(
        linear=double_stream_block.txt_mlp.layers[0],
        tensors=tensors,
        prefix=f"{prefix}.txt_mlp.0",
    )
    double_stream_block.txt_mlp.layers[2] = port_linear(
        linear=double_stream_block.txt_mlp.layers[2],
        tensors=tensors,
        prefix=f"{prefix}.txt_mlp.2",
    )

    return double_stream_block


def port_single_stream_block(single_stream_block, tensors, prefix):
    single_stream_block.linear1 = port_linear(
        linear=single_stream_block.linear1, tensors=tensors, prefix=f"{prefix}.linear1"
    )
    single_stream_block.linear2 = port_linear(
        linear=single_stream_block.linear2, tensors=tensors, prefix=f"{prefix}.linear2"
    )

    single_stream_block.norm = port_qk_norm(
        qk_norm=single_stream_block.norm, tensors=tensors, prefix=f"{prefix}.norm"
    )

    # single_stream_block.pre_norm has no params

    single_stream_block.modulation = port_modulation(
        modulation=single_stream_block.modulation,
        tensors=tensors,
        prefix=f"{prefix}.modulation",
    )

    return single_stream_block


def port_mlp_embedder(mlp_embedder, tensors, prefix):
    mlp_embedder.in_layer = port_linear(
        linear=mlp_embedder.in_layer, tensors=tensors, prefix=f"{prefix}.in_layer"
    )

    mlp_embedder.out_layer = port_linear(
        linear=mlp_embedder.out_layer, tensors=tensors, prefix=f"{prefix}.out_layer"
    )
    return mlp_embedder


def port_final_layer(final_layer, tensors, prefix):
    # last_layer.norm_final has no params
    final_layer.linear = port_linear(
        linear=final_layer.linear,
        tensors=tensors,
        prefix=f"{prefix}.linear",
    )

    final_layer.adaLN_modulation.layers[1] = port_linear(
        linear=final_layer.adaLN_modulation.layers[1],
        tensors=tensors,
        prefix=f"{prefix}.adaLN_modulation.1",
    )

    return final_layer


def port_flux(flux, tensors):
    flux.img_in = port_linear(
        linear=flux.img_in,
        tensors=tensors,
        prefix="img_in",
    )

    flux.time_in = port_mlp_embedder(
        mlp_embedder=flux.time_in,
        tensors=tensors,
        prefix="time_in",
    )

    flux.vector_in = port_mlp_embedder(
        mlp_embedder=flux.vector_in,
        tensors=tensors,
        prefix="vector_in",
    )

    if flux.params.guidance_embed:
        flux.guidance_in = port_mlp_embedder(
            mlp_embedder=flux.guidance_in,
            tensors=tensors,
            prefix="guidance_in",
        )

    flux.txt_in = port_linear(
        linear=flux.txt_in,
        tensors=tensors,
        prefix="txt_in",
    )

    for i, layer in enumerate(flux.double_blocks.layers):
        layer = port_double_stream_block(
            double_stream_block=layer,
            tensors=tensors,
            prefix=f"double_blocks.{i}",
        )

    for i, layer in enumerate(flux.single_blocks.layers):
        layer = port_single_stream_block(
            single_stream_block=layer,
            tensors=tensors,
            prefix=f"single_blocks.{i}",
        )

    flux.final_layer = port_final_layer(
        final_layer=flux.final_layer,
        tensors=tensors,
        prefix="final_layer",
    )

    return flux
