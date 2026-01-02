"""T5 language model implementation."""

from typing import (
    Any,
    Dict,
    Optional,
)

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from audiocaption.core.base import LanguageModel
from audiocaption.core.registry import ModelRegistry


@ModelRegistry.register_language_model("t5")
class T5LanguageModel(LanguageModel):
    """T5 language model for text generation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        model_name = config.get("model_name", "t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

        if self.freeze:
            self.freeze_weights()
        else:
            self.unfreeze_weights()

    def get_embedding_dim(self) -> int:
        """Return T5 embedding dimension."""
        return self.model.config.d_model

    def get_tokenizer(self):
        """Return the T5 tokenizer."""
        return self.tokenizer
    
    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get T5 text embeddings from input token IDs."""
        return self.model.shared(input_ids)

    def forward_with_prefix(
        self,
        prefix_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with prefix embeddings.

        For T5, we use the prefix as encoder input and text embeddings as decoder input.

        Args:
            prefix_embeddings: Tensor of shape (batch_size, prefix_len, embedding_dim)
            text_embeddings: Tensor of shape (batch_size, seq_len, embedding_dim)
            labels: Optional labels tensor of shape (batch_size, seq_len)

        Returns:
            Dictionary with 'loss' key if labels provided
        """
        batch_size = prefix_embeddings.shape[0]

        # Use prefix embeddings as encoder input
        encoder_outputs = self.model.encoder(inputs_embeds=prefix_embeddings)

        # Use text embeddings as decoder input
        decoder_inputs_embeds = text_embeddings

        if labels is not None:
            # T5 expects labels to be the target sequence (not shifted)
            # Create decoder attention mask (all ones for now, can be improved)
            decoder_attention_mask = torch.ones(
                (batch_size, text_embeddings.shape[1]),
                device=self.device,
                dtype=torch.long,
            )

            # For T5, labels should match the decoder input length
            # If labels include prefix, we need to adjust
            if labels.shape[1] > text_embeddings.shape[1]:
                # Labels include prefix, remove it
                labels = labels[:, -text_embeddings.shape[1] :]

            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )

            return {"loss": outputs.loss}
        else:
            # Inference mode
            decoder_attention_mask = torch.ones(
                (batch_size, text_embeddings.shape[1]),
                device=self.device,
                dtype=torch.long,
            )
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
            )
            return {}
