import torch
from torch.nn import CrossEntropyLoss
from typing import Any, Optional, Tuple, Union, List
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    BaseModelOutput,
)

from ....utils._logger import logger, WarningType
import transformers

try:
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_causal_attention_mask,
    )

    if hasattr(transformers.models, "mixtral"):
        from transformers.models.mixtral.modeling_mixtral import (
            load_balancing_loss_func,
        )
    from transformers.modeling_outputs import (
        MoeCausalLMOutputWithPast,
        MoeModelOutputWithPast,
    )
except ImportError:
    pass

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

# https://github.com/huggingface/transformers/blob/b647acdb53d251cec126b79e505bac11821d7c93/src/transformers/models/t5/modeling_t5.py#L1336  # noqa: B950
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


def GPTJForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )
    hidden_states = transformer_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.transformer.first_device)
        hidden_states = hidden_states.to(self.lm_head.weight.device)

    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]

    # make sure sampling in fp16 works correctly and
    # compute loss in fp32 to match with mesh-tf version
    # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
    lm_logits = self.lm_head(hidden_states).to(torch.float32)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        loss = loss.to(hidden_states.dtype)

    output = (lm_logits,) + transformer_outputs[1:]
    return ((loss,) + output) if loss is not None else output


def LlamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if hasattr(self, "_prepare_decoder_attention_mask"):
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...",
                _type=WarningType.WrongArgument,
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def LlamaForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )

    hidden_states = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]

    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    output = (logits,) + outputs[1:]
    return (loss,) + output if loss is not None else output


def GPTNeoXForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    outputs = self.gpt_neox(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )

    hidden_states = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    lm_logits = self.embed_out(hidden_states)

    lm_loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )

    output = (lm_logits,) + outputs[1:]
    return ((lm_loss,) + output) if lm_loss is not None else output


def OPTForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
    prefill_policy: Optional[int] = None,
    decoding_policy: Optional[int] = None,
    no_overlap: Optional[bool] = None,
    pin_weight: Optional[bool] = None,
    gpu_percentage: Optional[int] = None,
    num_minibatch: Optional[int] = None,
    enable_cxl: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
        prefill_policy=prefill_policy,
        decoding_policy=decoding_policy,
        no_overlap=no_overlap,
        pin_weight=pin_weight,
        gpu_percentage=gpu_percentage,
        num_minibatch=num_minibatch,
        enable_cxl=enable_cxl,
        max_new_tokens=max_new_tokens,
    )
    hidden_states = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    logits = self.lm_head(hidden_states).contiguous()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
        )

    output = (logits,) + outputs[1:]
    return (loss,) + output if loss is not None else output


def BloomForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
    **deprecated_arguments,
) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    if deprecated_arguments.pop("position_ids", False) is not False:
        logger.warning(
            "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
            " passing `position_ids`.",
            _type=WarningType.DeprecatedArgument,
        )
    if len(deprecated_arguments) > 0:
        raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )
    hidden_states = transformer_outputs[0]

    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size),
            shift_labels.view(batch_size * seq_length),
        )

    output = (lm_logits,) + transformer_outputs[1:]
    return ((loss,) + output) if loss is not None else output


def FalconForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
    if position_ids is None:
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
    else:
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
    hidden_states = transformer_outputs[0]

    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size),
            shift_labels.view(batch_size * seq_length),
        )

    output = (lm_logits,) + transformer_outputs[1:]
    return ((loss,) + output) if loss is not None else output


def CodeGenForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """

    transformer_outputs = self.transformer(
        input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )
    hidden_states = transformer_outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    # make sure sampling in fp16 works correctly and
    # compute loss in fp32 to match with mesh-tf version
    # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
    lm_logits = self.lm_head(hidden_states).to(torch.float32)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        loss = loss.to(hidden_states.dtype)

    output = (lm_logits,) + transformer_outputs[1:]
    return ((loss,) + output) if loss is not None else output


def BaichuanForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = True,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    return_dict: Optional[bool] = False,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    if position_ids is not None:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
    else:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

    hidden_states = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        softmax_normalizer = shift_logits.max(-1).values ** 2
        z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) + z_loss

    output = (logits,) + outputs[1:]
    return (loss,) + output if loss is not None else output


def ChatGLMModel_forward(
    self,
    input_ids,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    full_attention_mask: Optional[torch.BoolTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
):
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if self.pre_seq_len is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(
                batch_size=batch_size,
                device=input_ids.device,
                dtype=inputs_embeds.dtype,
            )
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask.new_ones((batch_size, self.pre_seq_len)),
                    attention_mask,
                ],
                dim=-1,
            )

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (
            past_key_values and seq_length != 1
        ):
            full_attention_mask = self.get_masks(
                input_ids, past_key_values, padding_mask=attention_mask
            )

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    # if position_ids is not None:
    #     rotary_pos_emb = rotary_pos_emb[position_ids]
    # else:
    #     rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    # rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds,
        full_attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
    )

    return tuple(
        v
        for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
        if v is not None
    )


def GLMTransformer_forward(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_caches=None,
    use_cache: Optional[bool] = True,
    output_hidden_states: Optional[bool] = False,
):
    if not kv_caches:
        kv_caches = [None for _ in range(self.num_layers)]
    presents = () if use_cache else None
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...",
                _type=WarningType.WrongArgument,
            )
            use_cache = False

    all_self_attentions = None
    all_hidden_states = () if output_hidden_states else None
    for index in range(self.num_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer = self._get_layer(index)
        if self.gradient_checkpointing and self.training:
            layer_ret = torch.utils.checkpoint.checkpoint(
                layer,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_caches[index],
                use_cache,
            )
        else:
            layer_ret = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                kv_cache=kv_caches[index],
                use_cache=use_cache,
            )
        hidden_states, kv_cache = layer_ret
        if use_cache:
            presents = presents + (kv_cache,)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # Final layer norm.
    if self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states, presents, all_hidden_states, all_self_attentions


def ChatGLMForConditionalGeneration_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
    position_ids: Optional[torch.Tensor] = None,
    return_last_logit: Optional[bool] = False,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )

    hidden_states = transformer_outputs[0]
    if (
        return_last_logit
        and hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[-1:]
    lm_logits = self.transformer.output_layer(hidden_states)
    lm_logits = lm_logits.transpose(0, 1).contiguous()

    loss = None
    if labels is not None:
        lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)

    output = (lm_logits,) + transformer_outputs[1:]
    return ((loss,) + output) if loss is not None else output


def GPTBigCodeModel_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.Tensor]] = None,
    position_ids: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if batch_size <= 0:
        raise ValueError("batch_size has to be defined and > 0")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].shape[2]

    if (
        attention_mask is not None
        and len(attention_mask.shape) == 2
        and position_ids is None
    ):
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_length > 0:
            position_ids = position_ids[
                :, past_length : input_shape[-1] + past_length :
            ]
    elif position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Self-attention mask.
    query_length = input_shape[-1]
    key_length = past_length + query_length
    self_attention_mask = self.bias[
        None, key_length - query_length : key_length, :key_length
    ]

    if attention_mask is not None:
        self_attention_mask = self_attention_mask * attention_mask.view(
            batch_size, 1, -1
        ).to(dtype=torch.bool, device=self_attention_mask.device)

    # MQA models: (batch_size, query_length, n_heads, key_length)
    # MHA models: (batch_size, n_heads, query_length, key_length)
    # attention_mask = self_attention_mask.unsqueeze(2 if self.multi_query else 1)
    attention_mask = self_attention_mask.unsqueeze(1)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if (
        self.config.add_cross_attention
        and encoder_hidden_states is not None
        and encoder_attention_mask is not None
    ):
        if encoder_attention_mask.dim() == 2:
            encoder_attention_mask.unsqueeze(1)
        assert encoder_attention_mask.dim() == 3
        encoder_attention_mask = encoder_attention_mask.bool().unsqueeze(
            2 if self.multi_query else 1
        )
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = (
        () if output_attentions and self.config.add_cross_attention else None
    )
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (
                    outputs[3 if use_cache else 2],
                )

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    return tuple(
        v
        for v in [
            hidden_states,
            presents,
            all_hidden_states,
            all_self_attentions,
            all_cross_attentions,
        ]
        if v is not None
    )


def GPTBigCodeForCausalLM_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    position_ids: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
    r"""
    labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )
    hidden_states = transformer_outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    output = (lm_logits,) + transformer_outputs[1:]
    return ((loss,) + output) if loss is not None else output


def T5ForConditionalGeneration_forward(
    self,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    decoder_head_mask: Optional[torch.FloatTensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
        if self.config.num_layers == self.config.num_decoder_layers:
            logger.warning(
                __HEAD_MASK_WARNING_MSG, _type=WarningType.DeprecatedArgument
            )
            decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)

    if (
        labels is not None
        and decoder_input_ids is None
        and decoder_inputs_embeds is None
    ):
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)
        hidden_states = hidden_states.to(self.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder.first_device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(
                self.decoder.first_device
            )

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.encoder.first_device)
        sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
        sequence_output = sequence_output * (self.model_dim**-0.5)
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and sequence_output.size(1) != 1
    ):
        sequence_output = sequence_output[:, -1:, :]
    lm_logits = self.lm_head(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        # move labels to correct device to enable PP
        labels = labels.to(lm_logits.device)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

    output = (lm_logits,) + decoder_outputs[1:] + tuple(encoder_outputs)
    return ((loss,) + output) if loss is not None else output


def T5DenseGatedActDense_forward(self, hidden_states):
    hidden_gelu = self.act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    if (
        hasattr(self, "wo")
        and hasattr(self.wo, "weight")
        and isinstance(self.wo.weight, torch.Tensor)
        and hidden_states.dtype != self.wo.weight.dtype
        and self.wo.weight.dtype != torch.int8
    ):
        hidden_states = hidden_states.to(self.wo.weight.dtype)
    hidden_states = self.wo(hidden_states)
    return hidden_states


def T5DenseActDense_forward(self, hidden_states):
    hidden_states = self.wi(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.dropout(hidden_states)
    if (
        hasattr(self, "wo")
        and hasattr(self.wo, "weight")
        and isinstance(self.wo.weight, torch.Tensor)
        and hidden_states.dtype != self.wo.weight.dtype
        and self.wo.weight.dtype != torch.int8
    ):
        hidden_states = hidden_states.to(self.wo.weight.dtype)
    hidden_states = self.wo(hidden_states)
    return hidden_states


def MistralModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if (
        attention_mask is not None
        and hasattr(self.config, "_flash_attn_2_enabled")
        and self.config._flash_attn_2_enabled
        and past_key_values is not None
    ):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if hasattr(self, "_prepare_decoder_attention_mask"):
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...",
                _type=WarningType.WrongArgument,
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def MistralForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )

    hidden_states = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    output = (logits,) + outputs[1:]
    return (loss,) + output if loss is not None else output


def MptForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = True,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )
    hidden_states = transformer_outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size),
            shift_labels.view(batch_size * seq_length),
        )

    output = (lm_logits,) + transformer_outputs[1:]
    return ((loss,) + output) if loss is not None else output


def MixtralForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = False,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )

    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits if return_dict else outputs[-1],
            self.num_experts,
            self.num_experts_per_tok,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss
    output = (logits,) + outputs[1:]
    if output_router_logits:
        output = (aux_loss,) + output
    return (loss,) + output if loss is not None else output


def MixtralModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    past_key_values_length = 0
    seq_length_with_past = seq_length
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...",
                _type=WarningType.WrongArgument,
            )
            use_cache = False

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # 4d mask is passed through the layers
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
        sliding_window=self.config.sliding_window,
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                output_router_logits,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-1],)
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_router_logits,
            ]
            if v is not None
        )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
    )


def StableLMEpochModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if hasattr(self, "_prepare_decoder_attention_mask"):
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...",
                _type=WarningType.WrongArgument,
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def StableLMEpochForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = False

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    logits = self.lm_head(hidden_states).float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    output = (logits,) + outputs[1:]
    return (loss,) + output if loss is not None else output


def QWenModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))

    if attention_mask is not None:
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    hidden_states = inputs_embeds

    kv_seq_len = hidden_states.size()[1]
    if past_key_values[0] is not None:
        kv_seq_len += past_key_values[0][0].shape[1]

    if self.training or not self.use_dynamic_ntk:
        ntk_alpha_list = [1.0]
    elif kv_seq_len != hidden_states.size()[1]:
        ntk_alpha_list = self.rotary_emb._ntk_alpha_cached_list
    else:
        ntk_alpha_list = []
        if attention_mask is not None and kv_seq_len > self.seq_length:
            true_seq_lens = (
                attention_mask.squeeze(1)
                .squeeze(1)
                .eq(0)
                .sum(dim=-1, dtype=torch.int32)
            )
            for i in range(hidden_states.size()[0]):
                true_seq_len = true_seq_lens[i].item()
                ntk_alpha = self.get_ntk_alpha(true_seq_len)
                ntk_alpha_list.append(ntk_alpha)
        else:
            ntk_alpha = self.get_ntk_alpha(kv_seq_len)
            ntk_alpha_list.append(ntk_alpha)
    self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
    rotary_pos_emb_list = [
        self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
    ]

    hidden_states = self.drop(hidden_states)
    output_shape = input_shape + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...",
                _type=WarningType.WrongArgument,
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                rotary_pos_emb_list,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb=rotary_pos_emb_list,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, presents, all_hidden_states] if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def QWenLMHeadModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )
    hidden_states = transformer_outputs[0]

    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    output = (lm_logits,) + transformer_outputs[1:]
    return ((loss,) + output) if loss is not None else output


def GitEncoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    pixel_values_present: Optional[bool] = False,
    return_dict: Optional[bool] = True,
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...",
                _type=WarningType.WrongArgument,
            )
            use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    next_decoder_cache = () if use_cache else None
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                layer_module.__call__,
                hidden_states,
                attention_mask,
                layer_head_mask,
                past_key_value,
                output_attentions,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                pixel_values_present=pixel_values_present,
            )

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def GitForCausalLM_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.Tensor]] = None,
    pixel_values: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
    if labels is not None:
        use_cache = False

    outputs = self.git(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        pixel_values=pixel_values,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )

    sequence_output = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and sequence_output.size(1) != 1
    ):
        sequence_output = sequence_output[:, -1:, :]
    logits = self.output(sequence_output)

    loss = None
    if labels is not None:
        # we are doing next-token prediction; shift prediction scores and input ids by one
        num_image_tokens = self.git.encoder.layer[0].attention.self.image_patch_tokens
        shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shifted_logits.view(-1, self.config.vocab_size), labels.view(-1)
        )

    output = (logits,) + outputs[1:]
    return ((loss,) + output) if loss is not None else output


def GitVisionEncoder_forward(
    self,
    inputs_embeds,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutput]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    bias = None
    if causal_attention_mask is not None:
        bias += causal_attention_mask
    if attention_mask is not None:
        bias += attention_mask
    hidden_states = inputs_embeds
    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                encoder_layer.__call__,
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=bias,
                output_attentions=output_attentions,
                vision=True,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, encoder_states, all_attentions] if v is not None
        )
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=encoder_states,
        attentions=all_attentions,
    )


def GitModel_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length = input_shape[1]

    # past_key_values_length
    past_key_values_length = (
        past_key_values[0][0].shape[2] if past_key_values is not None else 0
    )

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    projected_visual_features = None
    if pixel_values is not None:
        if pixel_values.ndim == 4:
            # here we assume pixel_values is of shape (batch_size, num_channels, height, width)
            visual_features = self.image_encoder(pixel_values).last_hidden_state

        elif pixel_values.ndim == 5:
            # here we assume pixel_values is of shape (batch_size, num_frames, num_channels, height, width)
            visual_features = []
            for frame_idx in range(pixel_values.shape[1]):
                visual_features_frame = self.image_encoder(
                    pixel_values[:, frame_idx, :, :]
                ).last_hidden_state
                visual_features_frame += self.img_temperal_embedding[frame_idx]
                visual_features.append(visual_features_frame)

            # finally, concatenate all features along sequence dimension
            visual_features = torch.cat(visual_features, dim=1)

        else:
            raise ValueError("pixel_values must be of rank 4 or 5")

        projected_visual_features = self.visual_projection(visual_features)

    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )

    if projected_visual_features is None:
        projected_visual_features = torch.zeros(
            (embedding_output.shape[0], 0, embedding_output.shape[2]),
            dtype=embedding_output.dtype,
            device=embedding_output.device,
        )

    # Repeat visual features to match embedding batch size.
    projected_visual_features = projected_visual_features.repeat(
        embedding_output.size(0) // projected_visual_features.size(0), 1, 1
    )

    # concatenate patch token and text token embeddings
    hidden_states = torch.cat((projected_visual_features, embedding_output), dim=1)

    # By default, an additive causal mask is created
    # for masking the future (one direction).
    tgt_mask = self._generate_future_mask(
        seq_length, embedding_output.dtype, embedding_output.device
    )

    # Create an attention mask of shape (batch_size, 1, tgt_seq_len, src_seq_len)
    combined_attention_mask = self.create_attention_mask(
        tgt=embedding_output,
        memory=projected_visual_features,
        tgt_mask=tgt_mask,
        past_key_values_length=past_key_values_length,
    )

    encoder_outputs = self.encoder(
        hidden_states,
        attention_mask=combined_attention_mask,
        head_mask=head_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values_present=pixel_values is not None,
    )
    sequence_output = encoder_outputs[0]

    return (sequence_output,) + encoder_outputs[1:]


def CLIPEncoder_forward(
    self,
    inputs_embeds,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutput]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    hidden_states = inputs_embeds
    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                encoder_layer.__call__,
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                vision=True,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, encoder_states, all_attentions] if v is not None
        )
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=encoder_states,
        attentions=all_attentions,
    )


def LlavaLlamaForCausalLM_forward(
    self,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    images: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    input_ids: torch.LongTensor = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=False,
    )

    hidden_states = outputs[0]
    if (
        hasattr(self, "config")
        and hasattr(self.config, "lm_head_generation")
        and self.config.lm_head_generation
        and hidden_states.size(1) != 1
    ):
        hidden_states = hidden_states[:, -1:, :]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    output = (logits,) + outputs[1:]
    return (loss,) + output if loss is not None else output


def output_hook(module: torch.nn.Module, args, kwargs, outputs: Any):
    if module.config.use_return_dict or (
        "return_dict" in kwargs and kwargs["return_dict"]
    ):
        idx = 0
        loss = None
        aux_loss = None
        hidden_states = None
        attentions = None
        router_logits = None
        cross_attentions = None
        encoder_hidden_states = None
        encoder_attentions = None
        if "labels" in kwargs and kwargs["labels"]:
            loss = outputs[idx]
            idx += 1
        if "output_router_logits" in kwargs and kwargs["output_router_logits"]:
            aux_loss = outputs[idx]
            idx += 1
        logits = outputs[idx]
        idx += 1
        if idx < len(outputs):
            past_key_values = outputs[idx]
            idx += 1
        if (
            "output_hidden_states" in kwargs and kwargs["output_hidden_states"]
        ) or module.config.output_hidden_states:
            hidden_states = outputs[idx]
            idx += 1
        if (
            "output_attentions" in kwargs and kwargs["output_attentions"]
        ) or module.config.output_attentions:
            attentions = outputs[idx]
            idx += 1
            if idx < len(outputs):
                cross_attentions = outputs[idx]
                idx += 1
        if "output_router_logits" in kwargs and kwargs["output_router_logits"]:
            router_logits = outputs[idx]
            idx += 1
        if idx < len(outputs):
            last_hidden_state = outputs[idx]
            idx += 1
            if (
                "output_hidden_states" in kwargs and kwargs["output_hidden_states"]
            ) or module.config.output_hidden_states:
                encoder_hidden_states = outputs[idx]
                idx += 1
            if (
                "output_attentions" in kwargs and kwargs["output_attentions"]
            ) or module.config.output_attentions:
                encoder_attentions = outputs[idx]
                idx += 1
        if module.config.architectures[0] == "T5ForConditionalGeneration":
            return Seq2SeqLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                decoder_hidden_states=hidden_states,
                decoder_attentions=attentions,
                cross_attentions=cross_attentions,
                encoder_last_hidden_state=last_hidden_state,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attentions=encoder_attentions,
            )
        if module.config.architectures[0] == "MixtralForCausalLM":
            return MoeCausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                attentions=attentions,
                router_logits=router_logits,
            )

        if module.config.architectures[0] in [
            "BloomForCausalLM",
            "GPTBigCodeForCausalLM",
            "MptForCausalLM",
            "FalconForCausalLM",
            "RWForCausalLM",
        ]:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                attentions=attentions,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )
    return outputs


class IPEX_LLM_Model_Return(torch.nn.Module):
    def __init__(self, model, optimized_model):
        super().__init__()
        self.config = model.config
        self.optimized_model = optimized_model
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.optimized_model(*args, **kwargs)
        return output_hook(self.model, args, kwargs, outputs)

    def save(self, path):
        self.optimized_model.save(path)


def prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "attention_mask": attention_mask,
    }


def prepare_inputs_for_generation_llama(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    if past_key_values is not None:
        cache_length = past_length = past_key_values[0][0].shape[2]
        max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def prepare_inputs_for_generation_gptbigcode(
    self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
):
    token_type_ids = kwargs.get("token_type_ids", None)
    # Omit tokens covered by past_key_values
    if past_key_values:
        past_length = past_key_values[0][0].shape[2]

        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]
    else:
        position_ids = None

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )
    return model_inputs


def prepare_inputs_labels_for_multimodal_llavallama(
    self, input_ids, attention_mask, past_key_values, images, labels=None, **kwargs
):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if (
            past_key_values is not None
            and vision_tower is not None
            and images is not None
            and input_ids.shape[1] == 1
        ):
            attention_mask = torch.ones(
                (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        model_inputs = {
            "inputs_embeds": self.model.embed_tokens(input_ids).to(images[0].dtype),
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        if labels is not None:
            model_inputs["labels"] = labels
        return model_inputs

    if type(images) is list or images.ndim == 5:
        concat_images = torch.cat(list(image for image in images), dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        image_features = [x.flatten(0, 1) for x in image_features]
    else:
        image_features = self.encode_images(images)

    new_input_embeds = []
    new_labels = [] if labels is not None else None
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
            # multimodal LLM, but the current sample is not multimodal
            # FIXME: this is a hacky fix, for deepspeed zero3 to work
            half_len = cur_input_ids.shape[0] // 2
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
            cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
            cur_input_embeds = torch.cat(
                [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0
            )
            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        cur_new_input_embeds = []
        if labels is not None:
            cur_labels = labels[batch_idx]
            cur_new_labels = []
            assert cur_labels.shape == cur_input_ids.shape
        while image_token_indices.numel() > 0:
            cur_image_features = image_features[cur_image_idx]
            image_token_start = image_token_indices[0]
            if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                self.config, "mm_use_im_start_end", False
            ):
                cur_new_input_embeds.append(
                    self.get_model()
                    .embed_tokens(cur_input_ids[: image_token_start - 1])
                    .detach()
                )
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(
                        cur_input_ids[image_token_start - 1 : image_token_start]
                    )
                )
                cur_new_input_embeds.append(cur_image_features)
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(
                        cur_input_ids[image_token_start + 1 : image_token_start + 2]
                    )
                )
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_new_labels.append(
                        cur_labels[image_token_start : image_token_start + 1]
                    )
                    cur_labels = cur_labels[image_token_start + 2 :]
            else:
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                )
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_labels = cur_labels[image_token_start + 1 :]
            cur_image_idx += 1
            if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                self.config, "mm_use_im_start_end", False
            ):
                cur_input_ids = cur_input_ids[image_token_start + 2 :]
            else:
                cur_input_ids = cur_input_ids[image_token_start + 1 :]
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        if cur_input_ids.numel() > 0:
            if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                self.config, "mm_use_im_start_end", False
            ):
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids).detach()
                )
            else:
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids)
                )
            if labels is not None:
                cur_new_labels.append(cur_labels)
        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        new_input_embeds.append(cur_new_input_embeds)
        if labels is not None:
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

    if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        max_len = max(x.shape[0] for x in new_input_embeds)

        new_input_embeds_align = []
        for cur_new_embed in new_input_embeds:
            cur_new_embed = torch.cat(
                (
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                ),
                dim=0,
            )
            new_input_embeds_align.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

        if labels is not None:
            new_labels_align = []
            _new_labels = new_labels
            for cur_new_label in new_labels:
                cur_new_label = torch.cat(
                    (
                        cur_new_label,
                        torch.full(
                            (max_len - cur_new_label.shape[0],),
                            IGNORE_INDEX,
                            dtype=cur_new_label.dtype,
                            device=cur_new_label.device,
                        ),
                    ),
                    dim=0,
                )
                new_labels_align.append(cur_new_label)
            new_labels = torch.stack(new_labels_align, dim=0)

        if attention_mask is not None:
            new_attention_mask = []
            for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                attention_mask, _new_labels, new_labels
            ):
                new_attn_mask_pad_left = torch.ones(
                    (cur_new_labels.shape[0] - labels.shape[1],),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                new_attn_mask_pad_right = torch.zeros(
                    (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                cur_new_attention_mask = torch.cat(
                    (
                        new_attn_mask_pad_left,
                        cur_attention_mask,
                        new_attn_mask_pad_right,
                    ),
                    dim=0,
                )
                new_attention_mask.append(cur_new_attention_mask)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            assert attention_mask.shape == new_labels.shape
    else:
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)

        if attention_mask is not None:
            new_attn_mask_pad_left = torch.ones(
                (
                    attention_mask.shape[0],
                    new_input_embeds.shape[1] - input_ids.shape[1],
                ),
                dtype=torch.bool,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
            assert attention_mask.shape == new_input_embeds.shape[:2]
    model_inputs = {
        "inputs_embeds": new_input_embeds.to(images[0].dtype),
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
    }
    if new_labels is not None:
        model_inputs["labels"] = new_labels
    return model_inputs
