//! Streaming Test Helpers
//!
//! Utilities for creating realistic streaming chunks that simulate
//! how LLM tokens actually arrive (1-5 characters at a time).

/// Split input into realistic char-level chunks (2-3 chars each for determinism)
pub fn create_realistic_chunks(input: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
                                                                    
        let chunk_size = if i + 3 <= chars.len() && chars[i].is_ascii_alphanumeric() {
            3                                            
        } else {
            2                                         
        };

        let end = (i + chunk_size).min(chars.len());
        let chunk: String = chars[i..end].iter().collect();
        chunks.push(chunk);
        i = end;
    }

    chunks
}

/// Split input at strategic positions to test edge cases
/// This creates chunks that break at critical positions like after quotes, colons, etc.
pub fn create_strategic_chunks(input: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = input.chars().collect();

    for (i, &ch) in chars.iter().enumerate() {
        current.push(ch);

                                           
        let should_break = matches!(ch, '"' | ':' | ',' | '{' | '}' | '[' | ']')
            || (i > 0 && chars[i-1] == '"' && ch == ' ')                     
            || current.len() >= 5;                         

        if should_break && !current.is_empty() {
            chunks.push(current.clone());
            current.clear();
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

/// Create the bug scenario chunks: `{"name": "` arrives in parts
pub fn create_bug_scenario_chunks() -> Vec<&'static str> {
    vec![
        r#"{"#,
        r#"""#,
        r#"name"#,
        r#"""#,
        r#":"#,
        r#" "#,
        r#"""#,                                               
        r#"search"#,                       
        r#"""#,
        r#","#,
        r#" "#,
        r#"""#,
        r#"arguments"#,
        r#"""#,
        r#":"#,
        r#" "#,
        r#"{"#,
        r#"""#,
        r#"query"#,
        r#"""#,
        r#":"#,
        r#" "#,
        r#"""#,
        r#"test query"#,
        r#"""#,
        r#"}"#,
        r#"}"#,
    ]
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_realistic_chunks() {
        let input = r#"{"name": "test"}"#;
        let chunks = create_realistic_chunks(input);

                                      
        assert!(chunks.len() > 3);

                                              
        let reconstructed: String = chunks.join("");
        assert_eq!(reconstructed, input);
    }

    #[test]
    fn test_strategic_chunks_breaks_after_quotes() {
        let input = r#"{"name": "value"}"#;
        let chunks = create_strategic_chunks(input);

                                               
        assert!(chunks.iter().any(|c| c.ends_with('"')));
        assert!(chunks.iter().any(|c| c.ends_with(':')));

                                              
        let reconstructed: String = chunks.join("");
        assert_eq!(reconstructed, input);
    }

    #[test]
    fn test_bug_scenario_chunks() {
        let chunks = create_bug_scenario_chunks();
        let reconstructed: String = chunks.join("");

                                           
        assert!(reconstructed.contains(r#"{"name": "search""#));

                                                                                                      
        let joined = chunks.join("|");
        assert!(joined.contains(r#" |"#));                                            
    }
}
