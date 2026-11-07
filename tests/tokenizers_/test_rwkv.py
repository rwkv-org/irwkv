# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import RwkvTrie

from irwkv.tokenizers.rwkv import RwkvTokenizer
from irwkv.transformers_utils.rwkv import GenerationConfig, RwkvConfig, bytes_to_unicode


@pytest.fixture
def tokenizer():
    alphabet = bytes_to_unicode()
    vocab = {alphabet[byte]: byte + 1 for byte in range(256)}
    vocab.update({"<|endoftext|>": 0, "hello": 300, "world": 301, "he": 302})
    backend = Tokenizer(RwkvTrie(vocab))
    backend.decoder = ByteLevel()
    backend.add_special_tokens([AddedToken("<|endoftext|>", special=True)])
    return RwkvTokenizer(
        backend,
        {
            "eos_token": "<|endoftext|>",
            "chat_template": (
                "{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\n"
                "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
            ),
        },
    )


def test_longest_match_preserves_token_ids(tokenizer):
    assert tokenizer.encode("hello world") == [300, 33, 301]


@pytest.mark.parametrize(
    "text", ["你好🙂", "Cafe\u0301", "\x00\t\r\n", " hello  world "]
)
def test_bytes_round_trip_without_normalization(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_explicit(tokenizer):
    ids = tokenizer.encode("hello<|endoftext|>")
    assert ids == [300, 0]
    assert tokenizer.decode(ids, skip_special_tokens=True) == "hello"
    assert tokenizer.eos_token_id == 0


def test_truncation_does_not_change_shared_backend(tokenizer):
    assert tokenizer.encode("hello world", truncation=True, max_length=1) == [301]
    assert tokenizer.encode("hello world") == [300, 33, 301]


def test_chat_template_uses_native_tokens(tokenizer):
    messages = [{"role": "user", "content": "hello"}]
    text = tokenizer.apply_chat_template(
        conversation=messages, tokenize=False, add_generation_prompt=True
    )
    assert text == "user: hello\nassistant: "
    assert tokenizer.apply_chat_template(messages, add_generation_prompt=True) == (
        tokenizer.encode(text, add_special_tokens=False)
    )


def test_chat_template_kwargs_cannot_replace_json_messages(tokenizer):
    with pytest.raises(ValueError, match="cannot replace"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "original"}],
            messages=[{"role": "user", "content": "replacement"}],
        )


def test_continued_message_preserves_whitespace_and_original_json(tokenizer):
    messages = [{"role": "assistant", "content": " hello  "}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )
    assert text == "assistant:  hello  "
    assert messages == [{"role": "assistant", "content": " hello  "}]


def test_rwkv_uses_flower_instead_of_converted_eos_defaults():
    config = RwkvConfig(bos_token_id=0, eos_token_id=0)
    generation = GenerationConfig(eos_token_id=[0, 1], stop_strings=["\nUser:"])
    assert config.bos_token_id is None
    assert config.eos_token_id == generation.eos_token_id == 10060
    assert generation.stop_strings == ["✿"]


@pytest.mark.parametrize("style", ["bot", "assistant", "function_calling"])
@pytest.mark.parametrize("mode", ["fake_think", "open_think"])
def test_native_template_replays_complete_assistant_turns(tmp_path, style, mode):
    tokenizer = RwkvTokenizer.from_pretrained(tmp_path)
    question = {"role": "user", "content": "Q1\n\n  keep whitespace  "}
    answer = {"role": "assistant", "content": "A1\n\n  keep whitespace  "}
    if mode == "open_think":
        answer["reasoning"] = "R1"
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
        rwkv_prompt_template=style,
        rwkv_generation_prompt=mode,
    )
    prefix = tokenizer.apply_chat_template([question], **kwargs)
    history = tokenizer.apply_chat_template(
        [question, answer, {"role": "user", "content": "Q2"}], **kwargs
    )
    continuation = "R1</think>" if mode == "open_think" else ""
    assert history.startswith(prefix + continuation + answer["content"] + "✿")


def test_native_template_does_not_use_converted_chat_artifacts(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("legacy template")
    tokenizer = RwkvTokenizer.from_pretrained(tmp_path)
    messages = [{"role": "user", "content": "hello"}]
    assert tokenizer.apply_chat_template(messages, tokenize=False) == "User✿hello✿"
    assert (
        tokenizer.apply_chat_template(
            messages, tokenize=False, chat_template="{{ messages[0].content }}"
        )
        == "hello"
    )


def test_native_template_replays_thinking_returned_without_a_parser(tmp_path):
    tokenizer = RwkvTokenizer.from_pretrained(tmp_path)
    question = {"role": "user", "content": "Q1"}
    answer = {"role": "assistant", "content": "R1 \n</think> A1\n"}
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    prefix = tokenizer.apply_chat_template([question], **kwargs)
    history = tokenizer.apply_chat_template(
        [question, answer, {"role": "user", "content": "Q2"}], **kwargs
    )
    assert history.startswith(prefix + answer["content"] + "✿")


def test_native_tokenizer_can_be_shared_between_threads(tokenizer):
    texts = ["hello 你好 world", "he", "\n🙂"] * 16
    expected = [tokenizer.encode(text) for text in texts]
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(tokenizer.encode, texts)) == expected


def test_tokenizer_pickle_preserves_native_backend(tokenizer):
    restored = pickle.loads(pickle.dumps(tokenizer))
    assert restored.encode("hello 你好 world") == tokenizer.encode("hello 你好 world")


def test_incremental_decoding_preserves_utf8_and_stop_strings(tokenizer):
    from irwkv.sampling_params import SamplingParams
    from irwkv.v1.engine.detokenizer import IncrementalDetokenizer

    request = SimpleNamespace(
        request_id="native",
        prompt_token_ids=tokenizer.encode("user: "),
        sampling_params=SamplingParams(stop=["END"]),
    )
    decoder = IncrementalDetokenizer.from_new_request(tokenizer, request)
    deltas = []
    for token in tokenizer.encode("你好🙂 helloENDignored"):
        stop = decoder.update([token], stop_terminated=False)
        deltas.append(
            decoder.get_next_output_text(finished=stop is not None, delta=True)
        )
        if stop:
            break
    assert "".join(deltas) == "你好🙂 hello"


def test_native_artifact_loading_without_transformers(tmp_path, tokenizer):
    (tmp_path / "tokenizer.json").write_text("invalid converted tokenizer")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>", "chat_template": "{{ messages }}"})
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "rwkv", "hidden_size": 128, "num_hidden_layers": 2})
    )
    script = """
import importlib.abc
import sys
class NoTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'transformers' or fullname.startswith('transformers.'):
            raise AssertionError('Transformers was imported')
sys.meta_path.insert(0, NoTransformers())
from irwkv.tokenizers.rwkv import RwkvTokenizer
from irwkv.transformers_utils.config import get_config
from irwkv.config import IrwkvConfig, ModelConfig
from irwkv.renderers.registry import renderer_from_config
from irwkv.renderers.hf import safe_apply_chat_template
config = get_config(sys.argv[1], trust_remote_code=False)
tokenizer = RwkvTokenizer.from_pretrained(sys.argv[1])
assert config.hidden_size == 128 and config.intermediate_size == 512
assert tokenizer.decode(tokenizer.encode('hello world')) == 'hello world'
assert tokenizer.encode('<|endoftext|>') == [61, 125, 25258, 7588, 2318, 125, 63]
assert tokenizer.eos_token_id == 10060 and tokenizer.eos_token == '✿'
model_config = ModelConfig(model=sys.argv[1], dtype='bfloat16', max_model_len=64)
renderer = renderer_from_config(IrwkvConfig(model_config=model_config))
assert renderer.tokenizer.encode('hello world') == tokenizer.encode('hello world')
text = safe_apply_chat_template(model_config, renderer.tokenizer,
    [{'role': 'user', 'content': 'hello'}], tokenize=False)
assert 'hello' in text
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)


def test_original_vocabulary_loads_without_converted_files(tmp_path):
    tokenizer = RwkvTokenizer.from_pretrained(tmp_path)
    assert tokenizer.eos_token_id == 10060
    assert tokenizer.eos_token == "✿"
    assert tokenizer.decode([10060]) == "✿"
    assert tokenizer.decode([10060], skip_special_tokens=True) == ""
    assert tokenizer.vocab_size == 65529
    assert tokenizer.get_added_vocab() == {}
    assert tokenizer.all_special_ids == [10060]
    assert tokenizer.num_special_tokens_to_add() == 0
    assert tokenizer.encode("<think>") == [61, 35762, 63]
    assert tokenizer.encode("</think>") == [754, 35762, 63]
    assert tokenizer.encode("✿") == [10060]
    assert tokenizer.decode([0]) == ""


def test_rwkv_config_defaults_match_canonical_rank_shapes():
    config = RwkvConfig(hidden_size=4096, num_hidden_layers=32)
    assert (
        config.decay_low_rank_dim,
        config.a_low_rank_dim,
        config.v_low_rank_dim,
        config.gate_low_rank_dim,
    ) == (160, 160, 96, 320)


def test_chat_template_selection_and_continuation(tokenizer):
    tokenizer.chat_template = {
        "default": "{{ messages[-1].content }} END",
        "short": "short",
    }
    assert tokenizer.get_chat_template("short") == "short"
    assert (
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": "hello"}],
            continue_final_message=True,
            tokenize=False,
        )
        == "hello"
    )
    assert tokenizer.encode("hello", truncation=True, max_length=0) == []


def test_added_placeholder_updates_cached_vocabulary(tokenizer):
    assert tokenizer.add_special_tokens({"additional_special_tokens": ["<embed>"]}) == 1
    ids = tokenizer.encode("<embed>", add_special_tokens=False)
    assert len(ids) == 1 and ids[0] in tokenizer.all_special_ids
    assert tokenizer.get_vocab()["<embed>"] == ids[0]


@pytest.mark.parametrize(
    "required,single,text,accepted",
    [
        (False, False, "hello", True),
        (True, False, "hello", False),
        (False, False, '<tool>{"x":1}</tool>', True),
        (False, False, '<tool>{"x":"bad"}</tool>', False),
        (False, False, '<tool>{"x":1}</tool><tool>{"x":2}</tool>', True),
        (False, True, '<tool>{"x":1}</tool><tool>{"x":2}</tool>', False),
    ],
)
@pytest.mark.parametrize("original_vocab", [False, True])
def test_structural_constraints_without_transformers(
    tokenizer, required, single, text, accepted, original_vocab, tmp_path
):
    import llguidance

    from irwkv.tool_parsers.structural_tag import to_guidance_grammar
    from irwkv.v1.structured_output.backend_guidance import GuidanceBackend

    if original_vocab:
        tokenizer = RwkvTokenizer.from_pretrained(tmp_path)

    spec = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool>"],
            "at_least_one": required,
            "stop_after_first": single,
            "tags": [
                {
                    "type": "tag",
                    "begin": "<tool>",
                    "end": "</tool>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    }
    grammar = to_guidance_grammar(spec, llguidance.LLMatcher.grammar_from_json_schema)
    native = GuidanceBackend._build_ll_tokenizer(tokenizer, len(tokenizer))
    assert not llguidance.LLMatcher.validate_grammar(grammar, native)
    matcher = llguidance.LLMatcher(native, grammar)
    success = matcher.consume_tokens(tokenizer.encode(text))
    assert (success and matcher.is_accepting()) == accepted
