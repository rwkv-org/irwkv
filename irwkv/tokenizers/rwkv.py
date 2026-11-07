# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native byte tokenizer and chat-template adapter for RWKV artifacts."""

import ast
import contextlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from huggingface_hub.errors import EntryNotFoundError
from jinja2 import nodes
from jinja2.exceptions import TemplateError
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import RwkvTrie

from irwkv.transformers_utils.rwkv import (
    RWKV_EOS_TOKEN,
    RWKV_EOS_TOKEN_ID,
    bytes_to_unicode,
    load_json,
)

RWKV_VOCAB_PATH = Path(__file__).with_name("rwkv_vocab_v20230424.txt")
RWKV_CHAT_TEMPLATE_PATH = (
    Path(__file__).parents[1] / "transformers_utils/chat_templates/rwkv.jinja"
)


def load_rwkv_tokenizer() -> Tokenizer:
    """Build the native byte trie directly from the original RWKV vocabulary."""
    alphabet = bytes_to_unicode()
    vocab = {}
    for expected_id, line in enumerate(RWKV_VOCAB_PATH.read_text().splitlines(), 1):
        token_id, literal = line.split(" ", 1)
        literal, length = literal.rsplit(" ", 1)
        value = ast.literal_eval(literal)
        value = value.encode("utf-8") if isinstance(value, str) else value
        if int(token_id) != expected_id or len(value) != int(length):
            raise ValueError(f"Invalid RWKV vocabulary entry {token_id}")
        vocab["".join(alphabet[byte] for byte in value)] = expected_id
    if len(vocab) != 65529:
        raise ValueError("RWKV requires the original 65529-entry vocabulary")
    backend = Tokenizer(RwkvTrie(vocab))
    backend.decoder = ByteLevel()
    return backend


class BatchEncoding(dict):
    """Tokenized inputs accepted by the existing renderer interfaces."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def data(self):
        return self


class AssistantTracker(Extension):
    """Render generation blocks normally when no assistant mask is requested."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        call = self.call_method("_generation_support")
        return nodes.CallBlock(call, [], [], body).set_lineno(lineno)

    def _generation_support(self, caller):
        return caller()


@lru_cache(maxsize=32)
def _compile_chat_template(source: str):
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[AssistantTracker, "jinja2.ext.loopcontrols", "jinja2.ext.do"],
    )

    def raise_exception(message):
        raise TemplateError(message)

    environment.globals.update(
        raise_exception=raise_exception,
        strftime_now=lambda format: datetime.now().strftime(format),
    )
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
        value, **({"ensure_ascii": False} | kwargs)
    )
    return environment.from_string(source)


class RwkvTokenizer:
    """Use the native tokenizers backend directly, without Transformers."""

    is_fast = True
    padding_side = "left"

    def __init__(self, backend: Tokenizer, config: dict[str, Any], **kwargs):
        self._tokenizer = backend
        self.backend_tokenizer = backend
        self._vocab = backend.get_vocab()
        self._vocab_size = backend.get_vocab_size(with_added_tokens=False)
        self.max_token_id = max(self._vocab.values())
        self.max_chars_per_token = max(map(len, self._vocab))
        self.truncation_side = kwargs.get("truncation_side", "left")
        if self.truncation_side not in ("left", "right"):
            raise ValueError("truncation_side must be left or right")
        self.model_max_length = config.get("model_max_length", 2**63 - 1)
        self.name_or_path = kwargs.get("name_or_path", "")
        self.chat_template = config.get("chat_template")
        if isinstance(self.chat_template, list):
            self.chat_template = {
                item["name"]: item["template"] for item in self.chat_template
            }
        self.special_tokens_map = {}
        for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
            value = config.get(name)
            if isinstance(value, dict):
                value = value["content"]
            setattr(self, name, value)
            setattr(self, f"{name}_id", self._vocab.get(value))
            if value is not None:
                self.special_tokens_map[name] = value
        added = backend.get_added_tokens_decoder()
        self.all_special_ids = sorted(
            token_id for token_id, value in added.items() if value.special
        )
        self.all_special_tokens = [
            added[token_id].content for token_id in self.all_special_ids
        ]
        self._added_vocab = {
            value.content: token_id for token_id, value in added.items()
        }

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id,
        *args,
        revision=None,
        download_dir=None,
        trust_remote_code=False,
        **kwargs,
    ):
        local = Path(path_or_repo_id).exists()
        config = {}
        if not local or (Path(path_or_repo_id) / "tokenizer_config.json").is_file():
            with contextlib.suppress(EntryNotFoundError):
                config = load_json(
                    path_or_repo_id,
                    "tokenizer_config.json",
                    revision=revision,
                    cache_dir=download_dir,
                    token=kwargs.get("token"),
                )
        config = {
            key: value for key, value in config.items() if key == "model_max_length"
        }
        config["eos_token"] = RWKV_EOS_TOKEN
        config["chat_template"] = RWKV_CHAT_TEMPLATE_PATH.read_text()
        tokenizer = cls(
            load_rwkv_tokenizer(),
            config,
            name_or_path=str(path_or_repo_id),
            **kwargs,
        )
        if tokenizer.encode(RWKV_EOS_TOKEN) != [RWKV_EOS_TOKEN_ID]:
            raise ValueError("RWKV end-of-turn token does not match the vocabulary")
        tokenizer.eos_token_id = RWKV_EOS_TOKEN_ID
        tokenizer.all_special_ids = [RWKV_EOS_TOKEN_ID]
        tokenizer.all_special_tokens = [RWKV_EOS_TOKEN]
        return tokenizer

    def __len__(self):
        return self.max_token_id + 1

    @property
    def vocab_size(self):
        return self._vocab_size

    def get_vocab(self):
        return self._vocab

    def get_added_vocab(self):
        return self._added_vocab

    def num_special_tokens_to_add(self):
        return self._tokenizer.num_special_tokens_to_add(False)

    def add_special_tokens(self, tokens):
        values = tokens.get("additional_special_tokens", [])
        count = self._tokenizer.add_special_tokens(
            [AddedToken(value, special=True) for value in values]
        )
        self._vocab = self._tokenizer.get_vocab()
        self.max_token_id = max(self._vocab.values())
        self.max_chars_per_token = max(map(len, self._vocab))
        self._added_vocab = {
            value.content: token_id
            for token_id, value in self._tokenizer.get_added_tokens_decoder().items()
        }
        self.all_special_ids = sorted(
            token_id
            for token_id, value in self._tokenizer.get_added_tokens_decoder().items()
            if value.special
        )
        self.all_special_tokens = [
            self._tokenizer.id_to_token(i) for i in self.all_special_ids
        ]
        return count

    def encode(self, text, truncation=None, max_length=None, add_special_tokens=True):
        ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        if truncation:
            if max_length is None:
                max_length = self.model_max_length
            if max_length < 0:
                raise ValueError("max_length must be nonnegative")
            if max_length == 0:
                return []
            ids = (
                ids[-max_length:]
                if self.truncation_side == "left"
                else ids[:max_length]
            )
        return ids

    def __call__(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        if text_pair is not None:
            raise ValueError("RWKV does not use paired inputs")
        texts = [text] if isinstance(text, str) else text
        ids = [
            self.encode(value, truncation, max_length, add_special_tokens)
            for value in texts
        ]
        masks = [[1] * len(value) for value in ids]
        if isinstance(text, str):
            ids, masks = ids[0], masks[0]
        return BatchEncoding(input_ids=ids, attention_mask=masks)

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        if skip_special_tokens:
            ids = [token for token in ids if token not in self.all_special_ids]
        return self._tokenizer.decode(
            list(ids), skip_special_tokens=skip_special_tokens
        )

    def batch_decode(self, ids, skip_special_tokens=False, **kwargs):
        if skip_special_tokens:
            ids = [
                [token for token in row if token not in self.all_special_ids]
                for row in ids
            ]
        return self._tokenizer.decode_batch(
            ids, skip_special_tokens=skip_special_tokens
        )

    def batch_encode(self, texts, **kwargs):
        return self(texts, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            if tokens == self.eos_token:
                return self.eos_token_id
            return self._vocab.get(tokens, self.unk_token_id)
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            if ids == self.eos_token_id:
                return self.eos_token
            return self._tokenizer.id_to_token(ids)
        return [
            self.convert_ids_to_tokens(token_id)
            for token_id in ids
            if not skip_special_tokens or token_id not in self.all_special_ids
        ]

    def convert_tokens_to_string(self, tokens):
        return self.decode(self.convert_tokens_to_ids(tokens))

    def get_chat_template(self, chat_template=None, tools=None):
        template = self.chat_template
        if isinstance(template, dict):
            if chat_template in template:
                return template[chat_template]
            name = "tool_use" if tools and "tool_use" in template else "default"
            template = template.get(name)
        template = chat_template or template
        if not template:
            raise ValueError("No chat template is defined for this RWKV artifact")
        return template

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        chat_template=None,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=False,
        return_dict=False,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        messages = conversation
        if "messages" in kwargs:
            raise ValueError("Template kwargs cannot replace conversation messages")
        if continue_final_message and add_generation_prompt:
            raise ValueError(
                "continue_final_message conflicts with add_generation_prompt"
            )
        continuation_tag = "<|irwkv_continue_final_message|>"
        if continue_final_message:
            content = messages[-1]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("A continued message must contain nonempty text")
            messages = [dict(message) for message in messages]
            messages[-1]["content"] = content + continuation_tag
        template = self.get_chat_template(chat_template, tools)
        text = _compile_chat_template(template).render(
            **(
                self.special_tokens_map
                | {
                    "messages": messages,
                    "tools": tools,
                    "add_generation_prompt": add_generation_prompt,
                }
                | kwargs
            )
        )
        if continue_final_message:
            prefix, found, _ = text.rpartition(continuation_tag)
            if not found or content.strip() not in prefix:
                raise ValueError(
                    "The final message is absent from the rendered template"
                )
            text = prefix
        if not tokenize:
            return text
        encoded = self(
            text, add_special_tokens=False, truncation=truncation, max_length=max_length
        )
        return encoded if return_dict else encoded.input_ids
