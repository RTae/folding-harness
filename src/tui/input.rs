//! The message line: a small multi-line editor with a cursor, a history ring,
//! and the wrapping the transcript uses.
//!
//! `ratatui` draws widgets and leaves input to the application, so this is
//! hand-written. Chars are the unit throughout, so a multi-byte character
//! counts as one position and a paste of CJK text does not split mid-character.

/// Wrap text to a width, preserving explicit newlines. Done by hand so the
/// transcript's line count is known exactly and scrolling is predictable.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.trim().is_empty() {
            out.push(String::new());
            continue;
        }

        // Keep the line's own indentation and repeat it on continuations, so
        // nested JSON stays readable instead of collapsing to a flat block.
        let body = paragraph.trim_start();
        let depth = paragraph.chars().count() - body.chars().count();
        let indent = " ".repeat(depth.min(width.saturating_sub(8)));
        let width = width - indent.chars().count();

        let mut line = String::new();
        let push = |line: String, out: &mut Vec<String>| {
            out.push(format!("{indent}{line}"));
        };

        for word in body.split_whitespace() {
            if line.is_empty() {
                // A single word longer than the pane is hard-split.
                if word.chars().count() > width {
                    let mut rest = word;
                    while rest.chars().count() > width {
                        let head: String = rest.chars().take(width).collect();
                        push(head, &mut out);
                        rest = &rest[rest
                            .char_indices()
                            .nth(width)
                            .map(|(index, _)| index)
                            .unwrap_or(rest.len())..];
                    }
                    line.push_str(rest);
                } else {
                    line.push_str(word);
                }
            } else if line.chars().count() + 1 + word.chars().count() <= width {
                line.push(' ');
                line.push_str(word);
            } else {
                push(std::mem::take(&mut line), &mut out);
                line.push_str(word);
            }
        }
        if !line.is_empty() {
            push(line, &mut out);
        }
    }
    out
}

/// A single-line editor with a cursor and a history ring.
///
/// Hand-written because `ratatui` draws widgets and leaves input to the
/// application. Chars are the unit throughout, so a multi-byte character
/// counts as one position.
#[derive(Debug, Default)]
pub struct TextInput {
    chars: Vec<char>,
    cursor: usize,
    history: Vec<String>,
    /// Where we are when scrolling back; `None` means "editing a fresh line".
    recall: Option<usize>,
    /// The line being edited, stashed while scrolling through history.
    stashed: Option<Vec<char>>,
}

impl TextInput {
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn insert(&mut self, character: char) {
        self.chars
            .insert(self.cursor.min(self.chars.len()), character);
        self.cursor += 1;
    }

    /// Insert pasted text at the cursor, newlines and all.
    ///
    /// A paste arrives as one event when the terminal has bracketed paste, so
    /// its newlines are content rather than a run of Enter presses. One
    /// trailing newline is dropped: copying a line usually takes its line
    /// ending with it, and nobody means to end on a blank line.
    pub fn insert_str(&mut self, text: &str) {
        let text = text.strip_suffix('\n').unwrap_or(text);
        let text = text.strip_suffix('\r').unwrap_or(text);
        for character in text.chars() {
            // Carriage returns and tabs would corrupt the layout.
            match character {
                '\r' => continue,
                '\t' => {
                    for _ in 0..4 {
                        self.insert(' ');
                    }
                }
                _ => self.insert(character),
            }
        }
    }

    /// The logical lines of the buffer.
    pub fn lines(&self) -> Vec<String> {
        self.text()
            .split('\n')
            .map(|line| line.to_string())
            .collect()
    }

    pub fn line_count(&self) -> usize {
        1 + self.chars.iter().filter(|c| **c == '\n').count()
    }

    /// Where the cursor sits, as a line and a column within it.
    pub fn row_col(&self) -> (usize, usize) {
        let mut row = 0;
        let mut col = 0;
        for character in self.chars.iter().take(self.cursor) {
            if *character == '\n' {
                row += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (row, col)
    }

    /// Index of the first character of a given line.
    pub fn line_start(&self, row: usize) -> usize {
        let mut seen = 0;
        for (index, character) in self.chars.iter().enumerate() {
            if seen == row {
                return index;
            }
            if *character == '\n' {
                seen += 1;
            }
        }
        self.chars.len()
    }

    pub fn line_len(&self, row: usize) -> usize {
        self.lines()
            .get(row)
            .map(|line| line.chars().count())
            .unwrap_or(0)
    }

    /// Up and Down move between lines when the message has several, and
    /// otherwise walk the history, which is what a shell does.
    pub fn up(&mut self) {
        let (row, col) = self.row_col();
        if row == 0 {
            self.recall_older();
            return;
        }
        let target = row - 1;
        self.cursor = self.line_start(target) + col.min(self.line_len(target));
    }

    pub fn down(&mut self) {
        let (row, col) = self.row_col();
        if row + 1 >= self.line_count() {
            self.recall_newer();
            return;
        }
        let target = row + 1;
        self.cursor = self.line_start(target) + col.min(self.line_len(target));
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    /// Ctrl-W: delete the word before the cursor, including trailing spaces.
    pub fn delete_word(&mut self) {
        while self.cursor > 0 && self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
        while self.cursor > 0 && !self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.chars.len());
    }

    pub fn home(&mut self) {
        let (row, _) = self.row_col();
        self.cursor = self.line_start(row);
    }

    pub fn end(&mut self) {
        let (row, _) = self.row_col();
        self.cursor = self.line_start(row) + self.line_len(row);
    }

    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.recall = None;
        self.stashed = None;
    }

    /// Take the line, push it onto the history, and reset.
    pub fn take(&mut self) -> String {
        let text: String = self.chars.iter().collect();
        if !text.trim().is_empty() && self.history.last().map(String::as_str) != Some(text.as_str())
        {
            self.history.push(text.clone());
        }
        self.clear();
        text
    }

    pub fn set(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }

    /// Up: an older line from the history.
    pub fn recall_older(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.recall {
            None => {
                self.stashed = Some(self.chars.clone());
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(index) => index - 1,
        };
        self.recall = Some(next);
        let text = self.history[next].clone();
        self.set(&text);
    }

    /// Down: a newer line, then back to whatever was being typed.
    pub fn recall_newer(&mut self) {
        match self.recall {
            None => {}
            Some(index) if index + 1 < self.history.len() => {
                self.recall = Some(index + 1);
                let text = self.history[index + 1].clone();
                self.set(&text);
            }
            Some(_) => {
                self.recall = None;
                let stashed = self.stashed.take().unwrap_or_default();
                self.chars = stashed;
                self.cursor = self.chars.len();
            }
        }
    }

    /// The lines to draw and where the cursor sits among them.
    ///
    /// Only the cursor's own line scrolls horizontally, so the rest stay put
    /// while you move along a long line.
    pub fn view(&self, width: usize, max_rows: usize) -> (Vec<String>, usize, usize) {
        let width = width.max(4);
        let max_rows = max_rows.max(1);
        let lines = self.lines();
        let (row, col) = self.row_col();

        // Scroll vertically so the cursor's line is on screen.
        let first = if row >= max_rows {
            row + 1 - max_rows
        } else {
            0
        };
        let visible: Vec<String> = lines.iter().skip(first).take(max_rows).cloned().collect();

        let cursor_row = row - first;
        let mut shown = visible;
        let mut cursor_col = col;
        if let Some(line) = shown.get_mut(cursor_row) {
            if line.chars().count() >= width {
                let start = col.saturating_sub(width - 1);
                *line = line.chars().skip(start).take(width).collect();
                cursor_col = col - start;
            }
        }
        (shown, cursor_row, cursor_col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_respects_width_and_newlines() {
        let lines = wrap("one two three four five", 9);
        assert!(
            lines.iter().all(|line| line.chars().count() <= 9),
            "{lines:?}"
        );
        assert_eq!(wrap("a\n\nb", 20), vec!["a", "", "b"]);

        let long = wrap("QVQLVESGGGLVQAGGSLRLSCAAS", 10);
        assert!(
            long.iter().all(|line| line.chars().count() <= 10),
            "{long:?}"
        );
        assert_eq!(long.concat(), "QVQLVESGGGLVQAGGSLRLSCAAS");
    }

    #[test]
    fn wrapping_keeps_each_line_s_own_indentation() {
        // JSON in a reply is only readable if nesting survives the wrap.
        let json = "{\n  \"regions\": [\n    {\"name\": \"cdr1\"}\n  ]\n}";
        let lines = wrap(json, 40);
        assert_eq!(lines[0], "{");
        assert_eq!(lines[1], "  \"regions\": [");
        assert_eq!(lines[2], "    {\"name\": \"cdr1\"}");
        assert_eq!(lines[3], "  ]");
        assert_eq!(lines[4], "}");

        // A continuation of an indented line keeps the same indentation.
        let long = "    aaa bbb ccc ddd eee fff ggg hhh";
        let wrapped = wrap(long, 20);
        assert!(wrapped.len() > 1, "{wrapped:?}");
        assert!(
            wrapped.iter().all(|line| line.starts_with("    ")),
            "{wrapped:?}"
        );
        assert!(
            wrapped.iter().all(|line| line.chars().count() <= 20),
            "{wrapped:?}"
        );
    }

    #[test]
    fn the_input_line_edits_at_the_cursor() {
        let mut input = TextInput::default();
        for character in "hello".chars() {
            input.insert(character);
        }
        assert_eq!(input.text(), "hello");
        assert_eq!(input.cursor(), 5);

        input.left();
        input.left();
        input.insert('X');
        assert_eq!(input.text(), "helXlo");
        assert_eq!(input.cursor(), 4);

        input.home();
        assert_eq!(input.cursor(), 0);
        input.delete();
        assert_eq!(input.text(), "elXlo");
        input.end();
        input.backspace();
        assert_eq!(input.text(), "elXl");
    }

    #[test]
    fn ctrl_w_deletes_one_word_including_its_spaces() {
        let mut input = TextInput::default();
        for character in "design the loops   ".chars() {
            input.insert(character);
        }
        input.delete_word();
        assert_eq!(input.text(), "design the ");
        input.delete_word();
        assert_eq!(input.text(), "design ");
    }

    #[test]
    fn the_cursor_cannot_leave_the_line() {
        let mut input = TextInput::default();
        input.left();
        input.backspace();
        assert_eq!(input.cursor(), 0);
        assert_eq!(input.text(), "");
        input.insert('a');
        input.right();
        input.right();
        assert_eq!(input.cursor(), 1);
    }

    #[test]
    fn history_recalls_what_was_typed_and_comes_back() {
        let mut input = TextInput::default();
        input.set("first");
        assert_eq!(input.take(), "first");
        input.set("second");
        assert_eq!(input.take(), "second");

        input.set("draft");
        input.recall_older();
        assert_eq!(input.text(), "second");
        input.recall_older();
        assert_eq!(input.text(), "first");
        input.recall_older();
        assert_eq!(input.text(), "first", "stops at the oldest");
        input.recall_newer();
        assert_eq!(input.text(), "second");
        input.recall_newer();
        assert_eq!(input.text(), "draft", "returns to the unsent line");
    }

    #[test]
    fn a_blank_line_is_not_remembered_and_duplicates_collapse() {
        let mut input = TextInput::default();
        input.set("   ");
        input.take();
        input.set("same");
        input.take();
        input.set("same");
        input.take();
        assert_eq!(input.history, vec!["same"]);
    }

    #[test]
    fn a_paste_keeps_its_shape_and_drops_only_a_trailing_newline() {
        let mut input = TextInput::default();
        input.insert_str("a\nb\n");
        assert_eq!(input.text(), "a\nb");

        // Carriage returns are dropped and tabs become spaces, so neither
        // corrupts the layout.
        let mut windows = TextInput::default();
        windows.insert_str("one\r\ntwo\r\n");
        assert_eq!(windows.text(), "one\ntwo");

        let mut tabbed = TextInput::default();
        tabbed.insert_str("a\tb");
        assert_eq!(tabbed.text(), "a    b");
    }

    #[test]
    fn the_cursor_walks_a_multi_line_message() {
        let mut input = TextInput::default();
        input.insert_str("alpha\nbe\ngamma");
        assert_eq!(input.line_count(), 3);

        input.home();
        // Cursor is on the last line, at its start.
        assert_eq!(input.row_col(), (2, 0));

        input.up();
        assert_eq!(input.row_col(), (1, 0));
        input.end();
        assert_eq!(input.row_col(), (1, 2), "end of the short middle line");

        // Moving up from a short line clamps to the longer line's end.
        input.up();
        let (row, col) = input.row_col();
        assert_eq!(row, 0);
        assert!(col <= 5);

        input.down();
        assert_eq!(input.row_col().0, 1);
    }

    #[test]
    fn the_box_grows_with_the_message_and_stops_growing() {
        let mut input = TextInput::default();
        assert_eq!(input.line_count(), 1);
        input.insert_str(&"x\n".repeat(20));
        assert_eq!(input.line_count(), 20);

        // The view shows a window of the lines, not all twenty.
        let (lines, row, _) = input.view(40, 8);
        assert_eq!(lines.len(), 8);
        assert!(row < 8, "the cursor must stay on screen");
    }

    #[test]
    fn the_visible_window_follows_the_cursor() {
        let mut input = TextInput::default();
        for character in "abcdefghijklmnop".chars() {
            input.insert(character);
        }
        // At the end of a long line the tail is visible and the cursor sits
        // just past it — the cursor takes a column of its own, so the text is
        // at most width-1 characters.
        let (lines, _, column) = input.view(8, 1);
        let text = &lines[0];
        assert!(text.chars().count() <= 8, "{text:?}");
        assert!(text.ends_with('p'), "{text}");
        assert!(column < 8, "cursor column {column} is off-pane");
        assert_eq!(column, text.chars().count());

        input.home();
        let (lines, _, column) = input.view(8, 1);
        assert!(lines[0].starts_with('a'), "{:?}", lines[0]);
        assert_eq!(column, 0);

        // Somewhere in the middle, the cursor is still on-pane.
        for _ in 0..6 {
            input.right();
        }
        let (lines, _, column) = input.view(8, 1);
        assert!(column < 8, "cursor column {column} is off-pane");
        assert!(!lines[0].is_empty());
    }
}
