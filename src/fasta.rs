//! A minimal FASTA reader.
//!
//! Only what the harness needs: labelled amino-acid sequences. No wrapping
//! rules, no quality scores, no indexes. Keeping it here rather than taking a
//! dependency means the sequence path has no surprises in it.

use crate::scaffold::is_standard_aa;

#[derive(Debug, thiserror::Error)]
pub enum FastaError {
    #[error("could not read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{label}: '{residue}' is not a standard amino acid")]
    NonStandardResidue { label: String, residue: char },
    #[error("record '{label}' has no sequence")]
    EmptyRecord { label: String },
    #[error("no records found")]
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// First whitespace-delimited word of the header, or `chain<N>` when the
    /// header is bare.
    pub label: String,
    pub description: String,
    pub sequence: String,
}

/// Parse FASTA text. Sequences are upper-cased; `*` and `-` are dropped so a
/// stop codon or an alignment gap does not fail the residue check.
pub fn parse(text: &str) -> Result<Vec<Record>, FastaError> {
    let mut records: Vec<Record> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }
        if let Some(header) = line.strip_prefix('>') {
            let header = header.trim();
            let mut parts = header.splitn(2, char::is_whitespace);
            let first = parts.next().unwrap_or("").trim();
            let label = if first.is_empty() {
                format!("chain{}", records.len() + 1)
            } else {
                first.to_string()
            };
            records.push(Record {
                label,
                description: parts.next().unwrap_or("").trim().to_string(),
                sequence: String::new(),
            });
            continue;
        }

        // A bare sequence with no header at all is accepted as one record.
        if records.is_empty() {
            records.push(Record {
                label: "chain1".to_string(),
                description: String::new(),
                sequence: String::new(),
            });
        }
        if let Some(record) = records.last_mut() {
            for residue in line.chars() {
                if residue == '*' || residue == '-' || residue.is_whitespace() {
                    continue;
                }
                record.sequence.push(residue.to_ascii_uppercase());
            }
        }
    }

    if records.is_empty() {
        return Err(FastaError::Empty);
    }

    for record in &records {
        if record.sequence.is_empty() {
            return Err(FastaError::EmptyRecord {
                label: record.label.clone(),
            });
        }
        if let Some(residue) = record.sequence.chars().find(|c| !is_standard_aa(*c)) {
            return Err(FastaError::NonStandardResidue {
                label: record.label.clone(),
                residue,
            });
        }
    }

    Ok(records)
}

pub fn load(path: impl AsRef<std::path::Path>) -> Result<Vec<Record>, FastaError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| FastaError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse(&text)
}

/// Accept either a path to a FASTA file or a bare sequence typed on the command
/// line, so `--binder QVQL...` and `--binder binder.fasta` both work.
pub fn load_or_literal(value: &str, fallback_label: &str) -> Result<Vec<Record>, FastaError> {
    let path = std::path::Path::new(value);
    if path.exists() {
        return load(path);
    }
    let looks_like_a_sequence = !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '*' || c == '-');
    if looks_like_a_sequence {
        let mut records = parse(value)?;
        if let Some(first) = records.first_mut() {
            first.label = fallback_label.to_string();
        }
        return Ok(records);
    }
    // Not a file and not a sequence: report it as a missing file, which is
    // what the user most likely meant.
    Err(FastaError::Io {
        path: value.to_string(),
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "not a readable file, and not a bare amino-acid sequence",
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_come_from_the_first_header_word() {
        let records = parse(">H anti-target VHH\nQVQL\nVESG\n>A target\nMKTV\n").expect("parses");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].label, "H");
        assert_eq!(records[0].description, "anti-target VHH");
        assert_eq!(records[0].sequence, "QVQLVESG");
        assert_eq!(records[1].label, "A");
        assert_eq!(records[1].sequence, "MKTV");
    }

    #[test]
    fn lowercase_gaps_and_stops_are_normalised() {
        let records = parse(">x\nqvql-vesg*\n").expect("parses");
        assert_eq!(records[0].sequence, "QVQLVESG");
    }

    #[test]
    fn a_headerless_sequence_is_one_record() {
        let records = parse("QVQLVESG\n").expect("parses");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].label, "chain1");
    }

    #[test]
    fn a_non_standard_residue_is_refused_with_its_label() {
        let error = parse(">H\nQVQJ\n").expect_err("J is not standard");
        assert!(error.to_string().contains('H'), "{error}");
        assert!(error.to_string().contains('J'), "{error}");
    }

    #[test]
    fn an_empty_record_is_refused() {
        assert!(matches!(
            parse(">H\n>A\nMKTV\n"),
            Err(FastaError::EmptyRecord { .. })
        ));
    }

    #[test]
    fn a_literal_sequence_is_accepted_in_place_of_a_path() {
        let records = load_or_literal("QVQLVESG", "H").expect("parses");
        assert_eq!(records[0].label, "H");
        assert_eq!(records[0].sequence, "QVQLVESG");

        assert!(load_or_literal("./no/such/file.fasta", "H").is_err());
    }
}
