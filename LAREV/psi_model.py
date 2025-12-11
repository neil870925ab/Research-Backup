import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class Seq2SeqLeakProbe(nn.Module):
    """
    ψ model: frozen encoder + trainable decoder
    input: tilde_B (neutralized baseline)
    output: answer_text
    """
    def __init__(self, model_name_or_path="t5-large", tokenizer=None):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        if tokenizer is not None:
            self.model.resize_token_embeddings(len(tokenizer))
            self.model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<eos>")

        # 1. Keep the original shared embedding for the encoder (to freeze later)
        encoder_shared = self.model.shared

        # 2. Create a NEW embedding for the decoder
        vocab_size = self.model.config.vocab_size
        d_model = self.model.config.d_model

        decoder_embed = nn.Embedding(vocab_size, d_model)  
        decoder_embed.weight = nn.Parameter(encoder_shared.weight.clone()) # initialize from shared weights

        # 3. Plug the new embedding into the decoder
        self.model.decoder.set_input_embeddings(decoder_embed)

        # 4. Also re-link the lm_head to the new decoder embedding
        self.model.lm_head.weight = decoder_embed.weight

        # 5. Now freeze encoder and its original shared embedding
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in encoder_shared.parameters():
            p.requires_grad = False

        # 6. Allow decoder and its embedding + lm_head to train
        for p in self.model.decoder.parameters():
            p.requires_grad = True
        for p in decoder_embed.parameters():
            p.requires_grad = True
        for p in self.model.lm_head.parameters():
            p.requires_grad = True

        enc_trainable = sum(p.requires_grad for p in self.model.encoder.parameters())
        dec_trainable = sum(p.requires_grad for p in self.model.decoder.parameters())
        if enc_trainable != 0:
            raise ValueError("Encoder parameters should be frozen!")


    def forward(self, encoder_hidden_states, labels):
    # encoder_hidden_states: [B, L_enc, d], from Φ (regular model) encoder

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        output = self.model(
            encoder_outputs=encoder_outputs,
            labels=labels,
            return_dict=True,
        )
        return output.loss, output.logits