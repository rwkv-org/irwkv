// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use tokenizers::Tokenizer;
use tokenizers::decoders::byte_level::ByteLevel;
use tokenizers::models::rwkv_trie::RwkvTrie;

use crate::Result;

const VOCAB: &str = include_str!("../../../../irwkv/tokenizers/rwkv_vocab_v20230424.txt");
pub const RWKV_EOS_TOKEN_ID: u32 = 10060;

/// Load the original RWKV byte vocabulary without model conversion artifacts.
pub fn load_rwkv_tokenizer() -> Result<Tokenizer> {
    let mut alphabet = ['\0'; 256];
    let mut extra = 256;
    for (byte, character) in alphabet.iter_mut().enumerate() {
        let code = if (33..127).contains(&byte)
            || (161..173).contains(&byte)
            || (174..256).contains(&byte)
        {
            byte as u32
        } else {
            extra += 1;
            extra - 1
        };
        *character = char::from_u32(code).unwrap();
    }
    let mut entries = Vec::with_capacity(65529);
    for (index, line) in VOCAB.lines().enumerate() {
        let invalid = || tokenizer_error!("Invalid original RWKV vocabulary entry {}", index + 1);
        let (id, remainder) = line.split_once(' ').ok_or_else(invalid)?;
        let (literal, length) = remainder.rsplit_once(' ').ok_or_else(invalid)?;
        let id: u32 = id.parse().map_err(|_| invalid())?;
        let length: usize = length.parse().map_err(|_| invalid())?;
        let bytes = parse_literal(literal).ok_or_else(invalid)?;
        if id as usize != index + 1 || bytes.len() != length {
            return Err(invalid());
        }
        entries.push((
            bytes.iter().map(|&b| alphabet[b as usize]).collect::<String>(),
            id,
        ));
    }
    if entries.len() != 65529 {
        return Err(tokenizer_error!(
            "RWKV requires the original 65529-entry vocabulary"
        ));
    }
    let model = RwkvTrie::new(entries.into_iter().collect())
        .map_err(|error| tokenizer_error!("Invalid RWKV trie: {error}"))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    Ok(tokenizer)
}

// Original vocabulary lines contain Python str/bytes literals.
fn parse_literal(literal: &str) -> Option<Vec<u8>> {
    let (bytes, quoted) = match literal.strip_prefix('b') {
        Some(quoted) => (true, quoted),
        None => (false, literal),
    };
    let quote = quoted.chars().next()?;
    if !matches!(quote, '\'' | '"') || !quoted.ends_with(quote) || quoted.len() < 2 {
        return None;
    }
    let mut chars = quoted[1..quoted.len() - 1].chars();
    let mut result = Vec::new();
    while let Some(character) = chars.next() {
        let code = if character == '\\' {
            match chars.next()? {
                '\\' => 92,
                '\'' => 39,
                '"' => 34,
                'a' => 7,
                'b' => 8,
                'f' => 12,
                'n' => 10,
                'r' => 13,
                't' => 9,
                'v' => 11,
                escape @ ('x' | 'u' | 'U') => {
                    let count = match escape {
                        'x' => 2,
                        'u' if !bytes => 4,
                        'U' if !bytes => 8,
                        _ => return None,
                    };
                    let mut code = 0;
                    for _ in 0..count {
                        code = (code << 4) | chars.next()?.to_digit(16)?;
                    }
                    code
                }
                _ => return None,
            }
        } else {
            if bytes && !character.is_ascii() {
                return None;
            }
            character as u32
        };
        if bytes {
            result.push(code.try_into().ok()?);
        } else {
            let mut buffer = [0; 4];
            result.extend_from_slice(char::from_u32(code)?.encode_utf8(&mut buffer).as_bytes());
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_vocabulary_preserves_ids_without_synthetic_tokens() {
        let tokenizer = load_rwkv_tokenizer().unwrap();
        assert_eq!(tokenizer.get_vocab_size(true), 65529);
        assert!(tokenizer.get_added_tokens_decoder().is_empty());
        assert!(tokenizer.id_to_token(0).is_none());
        for (text, ids) in [
            ("<think>", vec![61, 35762, 63]),
            ("</think>", vec![754, 35762, 63]),
            ("✿", vec![10060]),
            ("<|endoftext|>", vec![61, 125, 25258, 7588, 2318, 125, 63]),
        ] {
            assert_eq!(tokenizer.encode(text, true).unwrap().get_ids(), ids);
        }
        let text = "中文🙂 Cafe\u{301}\0\t\r\n<think></think>✿<|endoftext|>";
        let ids = tokenizer.encode(text, false).unwrap();
        assert_eq!(tokenizer.decode(ids.get_ids(), false).unwrap(), text);
        assert_eq!(tokenizer.decode(&[0], false).unwrap(), "");
    }

    #[test]
    fn vocabulary_literals_preserve_raw_bytes_and_unicode() {
        assert_eq!(parse_literal(r"b'\xff\x00\n'"), Some(vec![255, 0, 10]));
        assert_eq!(
            parse_literal(r"'\x85\u200b\U0001f642'"),
            Some("\u{85}\u{200b}🙂".as_bytes().to_vec())
        );
        assert!(parse_literal(r"b'\u0041'").is_none());
        assert!(parse_literal(r"'\x0'").is_none());
    }
}
