use crate::expr::{Exp, Expr, Pattern};
use bumpalo::Bump;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::Peekable;
use std::str::Chars;

/// Represents the span of an error in the source code.
/// Line and column numbers are 1-indexed for Monaco editor.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ErrorSpan {
    pub start_line: usize,
    pub start_column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

/// Contains details about a parsing error.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ParseErrorDetails {
    pub message: String,
    pub span: Option<ErrorSpan>, // Span is optional for errors not tied to a specific location
}

impl fmt::Display for ParseErrorDetails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
        // Optionally, add span information if self.span.is_some()
        // For example:
        // if let Some(span) = &self.span {
        //     write!(f, "{} (at line {}, column {})", self.message, span.start_line, span.start_column)?;
        // } else {
        //     write!(f, "{}", self.message)?;
        // }
        // Ok(())
    }
}

// --- Position and Span ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    /// 0-indexed byte offset in the input string
    pub offset: usize,
    /// 1-indexed line number
    pub line: usize,
    /// 1-indexed column number (UTF-8 characters)
    pub column: usize,
}

impl Position {
    fn new(offset: usize, line: usize, column: usize) -> Self {
        Position {
            offset,
            line,
            column,
        }
    }

    fn advance(&mut self, char_value: char) {
        self.offset += char_value.len_utf8();
        if char_value == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

impl Span {
    fn new(start: Position, end: Position) -> Self {
        Span { start, end }
    }
}

// --- Custom Error for the parser ---
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    fn new(message: String, span: Span) -> Self {
        ParseError { message, span }
    }
}

// --- Token Definition ---

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind<'a> {
    Top,                    // T
    Assign,                 // :=
    Eq,                     // ==
    Lte,                    // <=
    Plus,                   // +
    And,                    // &
    Xor,                    // ^
    Minus,                  // -
    Tilde,                  // ~
    Not,                    // !
    Semicolon,              // ;
    Star,                   // *
    Dup,                    // dup
    LtlX,                   // X
    LtlU,                   // U
    LtlF,                   // F
    LtlG,                   // G
    LtlR,                   // R
    LParen,                 // (
    RParen,                 // )
    LBracket,               // [
    RBracket,               // ]
    DotDot,                 // ..
    Field(u32),             // x followed by digits
    Number(&'a str),        // Decimal numeric literal
    BinaryLiteral(&'a str), // Binary literal (0b1010)
    HexLiteral(&'a str),    // Hexadecimal literal (0xFF)
    IpLiteral(&'a str),     // IP address literal (192.168.1.1)
    Ident(&'a str),         // Variable identifier
    If,                     // if keyword
    Then,                   // then keyword
    Else,                   // else keyword
    Let,                    // let keyword
    In,                     // in keyword
    End,                    // end keyword (for multiple expressions parsing)
    Hole,                   // hole keyword (.nksynth hole declarations)
    Def,                    // def keyword (.nksynth definitions)
    Eof,                    // End of input
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenKind<'a>,
    pub span: Span,
}

impl<'a> Token<'a> {
    fn new(kind: TokenKind<'a>, span: Span) -> Self {
        Token { kind, span }
    }
}

// --- Lexer ---

pub struct Lexer<'a> {
    iter: Peekable<Chars<'a>>,
    current_pos: Position,
    input_str: &'a str,
    iterator_has_yielded_eof: bool, // New field to track EOF iteration state
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            iter: input.chars().peekable(),
            current_pos: Position::new(0, 1, 1),
            input_str: input,
            iterator_has_yielded_eof: false, // Initialize the new field
        }
    }

    fn next_char_with_pos(&mut self) -> Option<(char, Position)> {
        let start_pos = self.current_pos;
        match self.iter.next() {
            Some(c) => {
                self.current_pos.advance(c);
                Some((c, start_pos))
            }
            None => None,
        }
    }

    // Only peeks at char, doesn't advance position or return it
    fn peek_char(&mut self) -> Option<&char> {
        self.iter.peek()
    }

    fn parse_number_or_ip(&mut self, start_pos: Position) -> Result<TokenKind<'a>, ParseError> {
        while let Some(&next_c) = self.peek_char() {
            let literal = &self.input_str[start_pos.offset..self.current_pos.offset];
            if next_c.is_ascii_digit() {
                self.next_char_with_pos();
            } else if next_c == '.' {
                let mut lookahead = self.iter.clone();
                lookahead.next();
                if lookahead.peek() == Some(&'.') {
                    break;
                }
                self.next_char_with_pos();
            } else if next_c == '/' && literal.contains('.') {
                self.next_char_with_pos();
                while self.peek_char().is_some_and(|c| c.is_ascii_digit()) {
                    self.next_char_with_pos();
                }
                break;
            } else if next_c == '-' && literal.contains('.') {
                self.next_char_with_pos();
                while self
                    .peek_char()
                    .is_some_and(|c| c.is_ascii_digit() || *c == '.')
                {
                    self.next_char_with_pos();
                }
                break;
            } else {
                break;
            }
        }
        let literal = &self.input_str[start_pos.offset..self.current_pos.offset];
        if literal.contains('.') && self.is_valid_ip_format(literal) {
            Ok(TokenKind::IpLiteral(literal))
        } else {
            Ok(TokenKind::Number(literal))
        }
    }

    fn is_valid_ip_format(&self, s: &str) -> bool {
        // Handle CIDR notation (e.g., 192.168.1.0/24)
        let base_ip = if let Some(slash_pos) = s.find('/') {
            let (ip_part, suffix) = s.split_at(slash_pos);
            // Check if suffix is a valid CIDR prefix
            let prefix_part = &suffix[1..];
            if prefix_part.is_empty() || !prefix_part.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
            if let Ok(prefix) = prefix_part.parse::<u32>() {
                if prefix > 32 {
                    return false;
                }
            } else {
                return false;
            }
            ip_part
        } else if s.contains('-') {
            // Handle IP range (e.g., 192.168.1.10-192.168.1.20)
            let parts = s.split('-');
            if parts.clone().count() != 2 {
                return false;
            }
            // Validate both IPs in the range
            for ip in parts {
                if !self.is_valid_single_ip(ip) {
                    return false;
                }
            }
            return true;
        } else {
            s
        };

        self.is_valid_single_ip(base_ip)
    }

    fn is_valid_single_ip(&self, s: &str) -> bool {
        let parts = s.split('.');
        if parts.clone().count() != 4 {
            return false;
        }

        for part in parts {
            if part.is_empty() {
                return false;
            }
            // Check if all characters are digits
            if !part.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
            // Check if the number is in valid range (0-255)
            if let Ok(num) = part.parse::<u32>() {
                if num > 255 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    fn skip_whitespace(&mut self) -> Result<(), ParseError> {
        loop {
            let start_skip_pos = self.current_pos;
            match self.peek_char() {
                Some(&c) => {
                    if c == '/' {
                        // Potential comment
                        let mut temp_iter = self.iter.clone();
                        let mut temp_pos = self.current_pos;

                        temp_iter.next(); // Consume first '/'
                        temp_pos.advance('/');

                        if temp_iter.peek() == Some(&'/') {
                            // It's a line comment
                            self.next_char_with_pos(); // Consume first '/'
                            self.next_char_with_pos(); // Consume second '/'
                            while let Some(&next_c) = self.peek_char() {
                                if next_c == '\n' {
                                    // self.next_char_with_pos(); // Consume newline to advance position correctly
                                    break; // Stop before consuming newline, let outer loop handle it or EOF
                                }
                                self.next_char_with_pos(); // Consume comment char
                            }
                            continue; // Restart loop to check next char (could be newline or more whitespace)
                        } else {
                            // Single '/' is an error in NetKAT, but we treat it as unexpected here
                            // Or, let the main tokenizer handle it if '/' becomes an operator
                            return Err(ParseError::new(
                                "Unexpected character: /".to_string(),
                                Span::new(start_skip_pos, self.current_pos), // Span of just the '/'
                            ));
                        }
                    }
                    if !c.is_whitespace() {
                        break; // Not whitespace, not a comment
                    }
                    self.next_char_with_pos(); // Consume whitespace char
                }
                None => break, // EOF
            }
        }
        Ok(())
    }

    // This is the main tokenizing function that will need careful span management
    pub fn next_token(&mut self) -> Result<Token<'a>, ParseError> {
        self.skip_whitespace()?;

        let start_pos = self.current_pos;

        match self.next_char_with_pos() {
            None => Ok(Token::new(
                TokenKind::Eof,
                Span::new(start_pos, self.current_pos),
            )),
            Some((c, _char_start_pos)) => {
                // _char_start_pos is same as start_pos due to skip_whitespace
                let kind = match c {
                    // Handle '0' specially for 0b and 0x prefixes
                    '0' => {
                        match self.peek_char() {
                            Some(&'b') => {
                                // Binary literal
                                self.next_char_with_pos(); // Consume 'b'
                                let digits_start = self.current_pos.offset;
                                while let Some(&next_c) = self.peek_char() {
                                    if next_c == '0' || next_c == '1' {
                                        self.next_char_with_pos();
                                    } else {
                                        break;
                                    }
                                }
                                if self.current_pos.offset == digits_start {
                                    return Err(ParseError::new(
                                        "Expected binary digits after '0b'".to_string(),
                                        Span::new(start_pos, self.current_pos),
                                    ));
                                }
                                TokenKind::BinaryLiteral(
                                    &self.input_str[start_pos.offset + 2..self.current_pos.offset],
                                )
                            }
                            Some(&'x') | Some(&'X') => {
                                // Hexadecimal literal
                                self.next_char_with_pos(); // Consume 'x' or 'X'
                                let digits_start = self.current_pos.offset;
                                while let Some(&next_c) = self.peek_char() {
                                    if next_c.is_ascii_hexdigit() {
                                        self.next_char_with_pos();
                                    } else {
                                        break;
                                    }
                                }
                                if self.current_pos.offset == digits_start {
                                    return Err(ParseError::new(
                                        "Expected hexadecimal digits after '0x'".to_string(),
                                        Span::new(start_pos, self.current_pos),
                                    ));
                                }
                                TokenKind::HexLiteral(
                                    &self.input_str[start_pos.offset + 2..self.current_pos.offset],
                                )
                            }
                            _ => {
                                // Just a regular '0', continue to digit parsing
                                // Continue with the same logic as other digits
                                self.parse_number_or_ip(start_pos)?
                            }
                        }
                    }
                    '+' => TokenKind::Plus,
                    '&' => TokenKind::And,
                    '^' => TokenKind::Xor,
                    '-' => TokenKind::Minus,
                    '~' => TokenKind::Tilde,
                    '!' => TokenKind::Not,
                    ';' => TokenKind::Semicolon,
                    '*' => TokenKind::Star,
                    '(' => TokenKind::LParen,
                    ')' => TokenKind::RParen,
                    '[' => TokenKind::LBracket,
                    ']' => TokenKind::RBracket,
                    '.' => {
                        if self.peek_char() == Some(&'.') {
                            self.next_char_with_pos(); // Consume second '.'
                            TokenKind::DotDot
                        } else {
                            return Err(ParseError::new(
                                "Expected '..' for range operator".to_string(),
                                Span::new(start_pos, self.current_pos),
                            ));
                        }
                    }
                    ':' => {
                        if self.peek_char() == Some(&'=') {
                            self.next_char_with_pos(); // Consume '='
                            TokenKind::Assign
                        } else {
                            return Err(ParseError::new(
                                "Expected ':=' for assignment".to_string(),
                                Span::new(start_pos, self.current_pos),
                            ));
                        }
                    }
                    '=' => {
                        if self.peek_char() == Some(&'=') {
                            self.next_char_with_pos(); // Consume '='
                            TokenKind::Eq
                        } else {
                            // Single '=' is used in let bindings
                            TokenKind::Eq
                        }
                    }
                    '<' => {
                        if self.peek_char() == Some(&'=') {
                            self.next_char_with_pos(); // Consume '='
                            TokenKind::Lte
                        } else {
                            return Err(ParseError::new(
                                "Expected '<=' for inequality".to_string(),
                                Span::new(start_pos, self.current_pos),
                            ));
                        }
                    }
                    // Let all identifiers including 'x', 'x0', 'x1', etc. go through the general identifier handling
                    _ => {
                        // Check if it's a digit that could start a number
                        if c.is_ascii_digit() {
                            self.parse_number_or_ip(start_pos)?
                        }
                        // Check if it's a letter that could start an identifier
                        else if c.is_alphabetic() {
                            while let Some(&next_c) = self.peek_char() {
                                if next_c.is_alphanumeric() || next_c == '_' {
                                    self.next_char_with_pos();
                                } else {
                                    break;
                                }
                            }

                            // Check for reserved words
                            let ident = &self.input_str[start_pos.offset..self.current_pos.offset];
                            match ident {
                                "in" => TokenKind::In,
                                "if" => TokenKind::If,
                                "then" => TokenKind::Then,
                                "else" => TokenKind::Else,
                                "let" => TokenKind::Let,
                                "dup" => TokenKind::Dup,
                                "end" => TokenKind::End,
                                "hole" => TokenKind::Hole,
                                "def" => TokenKind::Def,
                                // Single letter tokens
                                "T" => TokenKind::Top,
                                "X" => TokenKind::LtlX,
                                "U" => TokenKind::LtlU,
                                "F" => TokenKind::LtlF,
                                "G" => TokenKind::LtlG,
                                "R" => TokenKind::LtlR,
                                // Handle field identifiers like x0, x1, x2, etc.
                                s if s.starts_with('x')
                                    && s.len() > 1
                                    && s[1..].chars().all(|c| c.is_ascii_digit()) =>
                                {
                                    match s[1..].parse::<u32>() {
                                        Ok(index) => TokenKind::Field(index),
                                        Err(_) => TokenKind::Ident(ident),
                                    }
                                }
                                _ => TokenKind::Ident(ident),
                            }
                        } else {
                            return Err(ParseError::new(
                                format!("Unexpected character: {}", c),
                                Span::new(start_pos, self.current_pos), // Span of the single unexpected char
                            ));
                        }
                    }
                };
                Ok(Token::new(kind, Span::new(start_pos, self.current_pos)))
            }
        }
    }
}

// --- Iterator for Lexer ---
// This now yields Result<Token<'a>, ParseError> where Token includes its span.
impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iterator_has_yielded_eof {
            return None; // Stop iteration if EOF has already been yielded
        }

        let token_result = self.next_token(); // Get the next token from the lexer's core logic

        match &token_result {
            // Peek into the result
            Ok(token) if token.kind == TokenKind::Eof => {
                self.iterator_has_yielded_eof = true; // Mark that EOF is about to be yielded
                Some(token_result) // Yield the EOF token
            }
            _ => {
                // For any other token or an error, just yield it.
                // If next_token() itself returns an error, that will be passed through.
                Some(token_result)
            }
        }
    }
}

// --- Parser ---
// This will need to be updated to use SpannedToken and ParseError

/*
OPERATOR PRECEDENCE AND ASSOCIATIVITY TABLE
===========================================

The parser implements the following precedence hierarchy (highest to lowest precedence):

1. PRIMARY EXPRESSIONS (highest precedence)
   - Literals: 0, 1, T, dup, end
   - Field operations: x1:=0, x2==1 (atomic constructs)
   - Parenthesized expressions: (expr)

2. POSTFIX OPERATORS
   - Star: *  (left-associative)
   - Example: a** = (a*)*

3. PREFIX OPERATORS
   - Complement: ~
   - Test Negation: !
   - LTL Next: X
   - LTL Future: F
   - LTL Globally: G
   - All prefix operators are right-associative
   - Example: ~~a = ~(~(a)), ~X a = ~(X(a)), !!a = !(!(a))

4. INTERSECTION (left-associative)
   - And: &
   - Example: a & b & c = ((a & b) & c)

5. SEQUENCE (left-associative)
   - Semicolon: ;
   - Example: a ; b ; c = ((a ; b) ; c)

6. ADDITIVE (left-associative)
   - Union: +
   - XOR: ^
   - Difference: -
   - Example: a + b - c = ((a + b) - c)

7. TEMPORAL OPERATORS (lowest precedence, right-associative)
   - Until: U
   - Release: R
   - Example: a U b U c = (a U (b U c))

PRECEDENCE INTERACTION EXAMPLES:
- !a* + b ; c & d U e  =>  (!(a*) + (b ; (c & d))) U e
- (a + b) ; c  =>  forces additive before sequence
- a ; b + c  =>  (a ; b) + c  (sequence binds tighter than additive)
- !a*  =>  !(a*)  (postfix binds tighter than prefix)
- a + b*  =>  a + (b*)  (postfix binds tighter than infix)

Field operations like x1:=0 and x2==1 are parsed as atomic expressions
at the primary level and have the highest precedence.
*/

pub struct Parser<'a> {
    // The lexer now yields Result<Token<'a>, ParseError>, where Token is SpannedToken
    lexer: Peekable<Lexer<'a>>,
    arena: &'a Bump,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>, arena: &'a Bump) -> Self {
        // Takes Lexer directly
        Parser {
            lexer: lexer.peekable(),
            arena,
        }
    }

    /// Peek at the next token, returning an owned clone.
    ///
    /// Exposed so clients (e.g. the `.nksynth` front-end in
    /// `crate::holes::parser`) can build a statement-level grammar on top of
    /// [`Parser::parse_single_expression`] while driving the same lexer.
    pub fn peek_cloned(&mut self) -> Result<Token<'a>, ParseError> {
        self.peek_token().cloned()
    }

    /// Consume and return the next token.
    pub fn advance(&mut self) -> Result<Token<'a>, ParseError> {
        self.consume_token()
    }

    /// Parses a single complete expression.
    pub fn parse_single_expression(&mut self) -> Result<Exp<'a>, ParseError> {
        // Returns ParseError
        let expr = self.parse_until()?; // Start with lowest precedence
        // Here, we might check for trailing tokens if a single expression is expected to consume all input.
        // For now, it parses one expression.
        Ok(expr)
    }

    // Helper to get the next token
    // Consumes the token from the lexer.
    fn next_token_internal(&mut self) -> Result<Token<'a>, ParseError> {
        // Returns full Token
        // The iterator's next() already wraps Lexer's next_token result.
        // If lexer.next() is None (because Eof was hit and made None), we synthesize an Eof token.
        // This ensures parser always has a token to look at, even if it's Eof.
        // However, the iterator implementation now makes Eof stop iteration (returns None).
        // So, if self.lexer.next() is None, it truly means the lexer is exhausted.
        // The lexer itself produces an Eof token before stopping.
        // This means the parser should get an Eof token as a valid item.

        // Let's reconsider: the lexer's iterator *will* stop if next_token returns an Eof Token.
        // To handle this, next_token_internal should perhaps check peek_token first.
        // If peek_token shows Eof, next_token_internal can return that Eof without advancing,
        // or advance and return it.
        // The current lexer iterator:
        // next() -> Some(Ok(Token(Eof, span))) -> then next call to next_token() from iterator.next() would be None.
        // So, the parser's next_token_internal will get Ok(Token(Eof, span))
        // And a subsequent call will get None from self.lexer.next().

        match self.lexer.next() {
            Some(Ok(token)) => Ok(token),
            Some(Err(e)) => Err(e),
            None => {
                // This case should ideally not be reached if the lexer guarantees an Eof token
                // before the iterator ends. If it can be reached, we need a dummy Eof span.
                // For now, assume Lexer's Eof is the definitive end.
                // If we get here, it's likely after an Eof has already been consumed.
                // Consider what span an Eof token generated here should have.
                // Perhaps the parser should primarily use peek_token and only consume when sure.
                // Let's rely on the lexer providing an Eof token.
                // This path implies the iterator is exhausted.
                // The parser's logic for parse_until etc. should handle TokenKind::Eof from peek_token.
                // So if next_token_internal is called when peek is Eof, it consumes Eof and returns it.
                // If called again, it would hit this None.
                // This feels like an unexpected state if the parser always checks peek_token.

                // Safest for now: if lexer iterator is exhausted, it means an EOF token was the last thing
                // it would have produced (or an error).
                // The `Parser::peek_token` should handle the `None` case from `self.lexer.peek()`
                // and map it to a reference to a synthetic EOF token.
                // `next_token_internal` should ideally not be called if `peek_token` is already EOF,
                // unless the parser logic explicitly wants to consume the EOF.

                // For now, let's assume this means the stream is truly finished.
                // The lexer's iterator gives None *after* it has given the EOF token.
                // So if the Parser calls next_token_internal *again* after receiving EOF, it gets None.
                Err(ParseError::new(
                    "Unexpected end of token stream after EOF".to_string(),
                    Default::default(),
                )) // Default span is not ideal
            }
        }
    }

    // Renaming to avoid conflict and signify it's the one the parser methods should use
    fn consume_token(&mut self) -> Result<Token<'a>, ParseError> {
        self.next_token_internal()
    }

    // Helper to peek at the next token without consuming it.
    // Returns a Result containing a reference to the Token or a cloned ParseError.
    // This now needs to handle the possibility of the lexer returning an error on peek.
    fn peek_token_internal(&mut self) -> Result<&Token<'a>, ParseError> {
        match self.lexer.peek() {
            Some(Ok(token_ref)) => Ok(token_ref),
            Some(Err(parse_error)) => Err(parse_error.clone()), // Clone error to return owned
            None => {
                // Iterator is exhausted. This implies an EOF token was already produced and consumed.
                // The parser logic should usually stop before this if it's peeking for next ops.
                // We need a way to represent EOF here without allocating a new Token each time.
                // This is tricky. For now, error. Later, perhaps a static EOF token reference.
                // Or, the lexer never truly "ends" but keeps yielding EOFs.
                // Current lexer: next_token yields Eof, then iterator's next yields None.
                // So, peek() will be None after Eof is consumed.
                Err(ParseError::new(
                    "Peeked beyond EOF".to_string(),
                    Default::default(),
                ))
            }
        }
    }

    // Renaming for clarity
    fn peek_kind(&mut self) -> Result<&TokenKind<'a>, ParseError> {
        self.peek_token_internal().map(|token| &token.kind)
    }

    fn peek_token(&mut self) -> Result<&Token<'a>, ParseError> {
        // Expose this one for parser methods
        self.peek_token_internal()
    }

    // Recursive descent parsing functions based on operator precedence:
    // New precedence hierarchy (highest to lowest):
    // 1. Postfix: *
    // 2. Prefix: !, X, F, G
    // 3. Intersection: & (left-associative)
    // 4. Sequence: ; (left-associative)
    // 5. Additive: +, ^, - (left-associative)
    // 6. Until/Release: U, R (right-associative)

    fn parse_until(&mut self) -> Result<Exp<'a>, ParseError> {
        let left = self.parse_additive()?;
        // Peek at the kind directly, to keep the token for its span if needed for error
        // This loop structure is for right-associativity. A single check is enough.
        match self.peek_token() {
            Ok(peeked_token) => {
                match peeked_token.kind {
                    TokenKind::LtlU => {
                        let op_token = self.consume_token()?; // Consume 'U'
                        // Check for EOF before parsing RHS
                        match self.peek_kind() {
                            Ok(&TokenKind::Eof) => Err(ParseError::new(
                                format!(
                                    "Expected expression after {} but found end of input",
                                    token_kind_to_user_string(&op_token.kind)
                                ),
                                op_token.span,
                            )),
                            Err(ref pe)
                                if pe.message.contains("Peeked beyond EOF")
                                    || pe.message.contains("Unexpected end of token stream") =>
                            {
                                Err(ParseError::new(
                                    format!(
                                        "Expected expression after {} but found end of input",
                                        token_kind_to_user_string(&op_token.kind)
                                    ),
                                    op_token.span,
                                ))
                            }
                            Ok(_) => {
                                // Other token, parse it
                                let right = self.parse_until()?;
                                Ok(Expr::ltl_until(self.arena, left, right))
                            }
                            Err(pe) => Err(pe.clone()), // Propagate other peek errors
                        }
                    }
                    TokenKind::LtlR => {
                        let op_token = self.consume_token()?; // Consume 'R'
                        // Check for EOF before parsing RHS
                        match self.peek_kind() {
                            Ok(&TokenKind::Eof) => Err(ParseError::new(
                                format!(
                                    "Expected expression after {} but found end of input",
                                    token_kind_to_user_string(&op_token.kind)
                                ),
                                op_token.span,
                            )),
                            Err(ref pe)
                                if pe.message.contains("Peeked beyond EOF")
                                    || pe.message.contains("Unexpected end of token stream") =>
                            {
                                Err(ParseError::new(
                                    format!(
                                        "Expected expression after {} but found end of input",
                                        token_kind_to_user_string(&op_token.kind)
                                    ),
                                    op_token.span,
                                ))
                            }
                            Ok(_) => {
                                // Other token, parse it
                                let right = self.parse_until()?;
                                let not_left = Expr::complement(self.arena, left);
                                let not_right = Expr::complement(self.arena, right);
                                let until = Expr::ltl_until(self.arena, not_left, not_right);
                                Ok(Expr::complement(self.arena, until))
                            }
                            Err(pe) => Err(pe.clone()), // Propagate other peek errors
                        }
                    }
                    _ => Ok(left), // No U or R, return current left
                }
            }
            Err(pe) => Err(pe.clone()), // Error peeking for U/R operator itself
        }
    }

    fn parse_additive(&mut self) -> Result<Exp<'a>, ParseError> {
        let mut left = self.parse_sequence()?;
        loop {
            // Peek at the kind directly, to keep the token for its span if needed for error
            let peeked_token_result = self.peek_token();

            match peeked_token_result {
                Ok(peeked_token) => {
                    match peeked_token.kind {
                        TokenKind::Plus => {
                            let op_token = self.consume_token()?; // Consume '+'
                            // Check for EOF before parsing RHS
                            if self.peek_kind()? == &TokenKind::Eof {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected expression after {} but found end of input",
                                        token_kind_to_user_string(&op_token.kind)
                                    ),
                                    op_token.span, // Span of the operator
                                ));
                            }
                            let right = self.parse_sequence()?;
                            left = Expr::union(self.arena, left, right);
                        }
                        TokenKind::Xor => {
                            let op_token = self.consume_token()?; // Consume '^'
                            // Check for EOF before parsing RHS
                            if self.peek_kind()? == &TokenKind::Eof {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected expression after {} but found end of input",
                                        token_kind_to_user_string(&op_token.kind)
                                    ),
                                    op_token.span,
                                ));
                            }
                            let right = self.parse_sequence()?;
                            left = Expr::xor(self.arena, left, right);
                        }
                        TokenKind::Minus => {
                            let op_token = self.consume_token()?; // Consume '-'
                            // Check for EOF before parsing RHS
                            if self.peek_kind()? == &TokenKind::Eof {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected expression after {} but found end of input",
                                        token_kind_to_user_string(&op_token.kind)
                                    ),
                                    op_token.span,
                                ));
                            }
                            let right = self.parse_sequence()?;
                            left = Expr::difference(self.arena, left, right);
                        }
                        _ => break, // Not Plus, Xor, or Minus
                    }
                }
                Err(pe) => return Err(pe.clone()), // Error during peeking
            }
        }
        Ok(left)
    }

    fn parse_sequence(&mut self) -> Result<Exp<'a>, ParseError> {
        let mut left = self.parse_intersect()?;
        while self.peek_kind()? == &TokenKind::Semicolon {
            let _op_token = self.consume_token()?; // Consume ';'
            let right = self.parse_intersect()?;
            left = Expr::sequence(self.arena, left, right);
        }
        Ok(left)
    }

    fn parse_intersect(&mut self) -> Result<Exp<'a>, ParseError> {
        let mut left = self.parse_prefix()?;
        while self.peek_kind()? == &TokenKind::And {
            self.consume_token()?; // Consume '&'
            let right = self.parse_prefix()?;
            left = Expr::intersect(self.arena, left, right);
        }
        Ok(left)
    }

    fn parse_prefix(&mut self) -> Result<Exp<'a>, ParseError> {
        match self.peek_kind()? {
            TokenKind::Tilde => {
                let op_token = self.consume_token()?; // Consume '~'
                // Check for EOF before parsing operand
                if self.peek_kind()? == &TokenKind::Eof {
                    return Err(ParseError::new(
                        format!(
                            "Expected expression after {} but found end of input",
                            token_kind_to_user_string(&op_token.kind)
                        ),
                        op_token.span,
                    ));
                }
                let operand = self.parse_prefix()?; // Right-associative
                Ok(Expr::complement(self.arena, operand))
            }
            TokenKind::Not => {
                let op_token = self.consume_token()?; // Consume '!'
                // Check for EOF before parsing operand
                if self.peek_kind()? == &TokenKind::Eof {
                    return Err(ParseError::new(
                        format!(
                            "Expected expression after {} but found end of input",
                            token_kind_to_user_string(&op_token.kind)
                        ),
                        op_token.span,
                    ));
                }
                let operand = self.parse_prefix()?; // Right-associative
                // We'll validate that operand is in test fragment during desugaring
                Ok(Expr::test_negation(self.arena, operand))
            }
            TokenKind::LtlX => {
                // LTL Next
                let op_token = self.consume_token()?; // Consume 'X'
                // Check for EOF before parsing operand
                if self.peek_kind()? == &TokenKind::Eof {
                    return Err(ParseError::new(
                        format!(
                            "Expected expression after {} but found end of input",
                            token_kind_to_user_string(&op_token.kind)
                        ),
                        op_token.span,
                    ));
                }
                let operand = self.parse_prefix()?;
                Ok(Expr::ltl_next(self.arena, operand))
            }
            TokenKind::LtlF => {
                // LTL Future: F e ≡ T U e
                let op_token = self.consume_token()?;
                // Check for EOF before parsing operand
                if self.peek_kind()? == &TokenKind::Eof {
                    return Err(ParseError::new(
                        format!(
                            "Expected expression after {} but found end of input",
                            token_kind_to_user_string(&op_token.kind)
                        ),
                        op_token.span,
                    ));
                }
                let operand = self.parse_prefix()?;
                Ok(Expr::ltl_until(self.arena, Expr::top(self.arena), operand)) // Uses helper from expr.rs
            }
            TokenKind::LtlG => {
                // LTL Globally: G e ≡ ¬F¬e ≡ ¬(T U ¬e)
                let op_token = self.consume_token()?;
                // Check for EOF before parsing operand
                if self.peek_kind()? == &TokenKind::Eof {
                    return Err(ParseError::new(
                        format!(
                            "Expected expression after {} but found end of input",
                            token_kind_to_user_string(&op_token.kind)
                        ),
                        op_token.span,
                    ));
                }
                let operand = self.parse_prefix()?;
                Ok(Expr::ltl_globally(self.arena, operand)) // Uses helper from expr.rs
            }
            _ => {
                // No prefix operator, parse postfix
                self.parse_postfix()
            }
        }
    }

    fn parse_postfix(&mut self) -> Result<Exp<'a>, ParseError> {
        let mut base_expr = self.parse_atom_or_field_expression()?;

        // Handle postfix star (left-associative, though only one postfix operator exists)
        while self.peek_kind()? == &TokenKind::Star {
            let _star_token = self.consume_token()?; // Consume '*'
            base_expr = Expr::star(self.arena, base_expr);
        }
        Ok(base_expr)
    }

    // This function replaces the old parse_primary_and_assign_eq and parts of parse_primary.
    // It parses atomic literals, dup, parenthesized expressions, and field assignments/tests.
    fn parse_atom_or_field_expression(&mut self) -> Result<Exp<'a>, ParseError> {
        let current_token = self.peek_token()?.clone(); // Clone to avoid lifetime issues if we consume

        match current_token.kind {
            TokenKind::Field(index) => {
                self.consume_token()?; // Consume field token

                let next_token_after_field = self.peek_token()?.clone();
                match next_token_after_field.kind {
                    TokenKind::LBracket => {
                        // Parse bit range: x[start..end]
                        self.consume_token()?; // Consume '['

                        // Parse start index
                        let start_token = self.consume_token()?;
                        let start = match start_token.kind {
                            TokenKind::Number(num_str) => num_str.parse::<u32>().map_err(|_| {
                                ParseError::new(
                                    format!("Invalid start index in bit range: {}", num_str),
                                    start_token.span,
                                )
                            })?,
                            _ => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected number for bit range start, found {}",
                                        token_kind_to_user_string(&start_token.kind)
                                    ),
                                    start_token.span,
                                ));
                            }
                        };

                        // Expect '..'
                        match self.consume_token()? {
                            Token {
                                kind: TokenKind::DotDot,
                                ..
                            } => {}
                            tok => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected '..' in bit range, found {}",
                                        token_kind_to_user_string(&tok.kind)
                                    ),
                                    tok.span,
                                ));
                            }
                        }

                        // Parse end index
                        let end_token = self.consume_token()?;
                        let end = match end_token.kind {
                            TokenKind::Number(num_str) => num_str.parse::<u32>().map_err(|_| {
                                ParseError::new(
                                    format!("Invalid end index in bit range: {}", num_str),
                                    end_token.span,
                                )
                            })?,
                            _ => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected number for bit range end, found {}",
                                        token_kind_to_user_string(&end_token.kind)
                                    ),
                                    end_token.span,
                                ));
                            }
                        };

                        // Expect ']'
                        match self.consume_token()? {
                            Token {
                                kind: TokenKind::RBracket,
                                ..
                            } => {}
                            tok => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected ']' to close bit range, found {}",
                                        token_kind_to_user_string(&tok.kind)
                                    ),
                                    tok.span,
                                ));
                            }
                        }

                        // Check what follows: := or ==
                        let op_token = self.peek_token()?.clone();
                        match op_token.kind {
                            TokenKind::Assign => {
                                self.consume_token()?; // Consume ':='

                                // Parse the number value
                                let value_token = self.consume_token()?;
                                let value_str = match value_token.kind {
                                    TokenKind::Number(num_str) => num_str,
                                    _ => {
                                        return Err(ParseError::new(
                                            format!(
                                                "Expected number after bit range assignment, found {}",
                                                token_kind_to_user_string(&value_token.kind)
                                            ),
                                            value_token.span,
                                        ));
                                    }
                                };

                                // Convert number to bit vector
                                let bits =
                                    number_to_bits(self.arena, value_str, (end - start) as usize)?;
                                Ok(self.arena.alloc(Expr::BitRangeAssign(start, end, bits)))
                            }
                            TokenKind::Tilde => {
                                // Pattern match operator x[start..end] ~ pattern
                                self.consume_token()?; // Consume '~'

                                // Parse pattern with field width
                                let field_width = (end - start) as usize;
                                let pattern = self.parse_pattern(Some(field_width))?;
                                Ok(Expr::bit_range_match(self.arena, start, end, pattern))
                            }
                            _ => Err(ParseError::new(
                                format!(
                                    "Expected ':=' or '~' after bit range, found {}",
                                    token_kind_to_user_string(&op_token.kind)
                                ),
                                op_token.span,
                            )),
                        }
                    }
                    TokenKind::Assign => {
                        self.consume_token()?; // Consume ':='

                        // Peek for the value token (0 or 1)
                        let value_token_peek = self.peek_token().map_err(|e| e.clone())?;
                        if value_token_peek.kind == TokenKind::Eof {
                            return Err(ParseError::new(
                                format!(
                                    "Expected '0' or '1' after 'x{} :=' but found end of input",
                                    index
                                ),
                                value_token_peek.span, // Span of the EOF token
                            ));
                        }
                        // If not EOF, proceed to parse it (original logic)
                        let rhs_expr = self.parse_primary()?; // Use parse_primary for 0 or 1
                        let value_bool = match *rhs_expr {
                            Expr::Zero => false,
                            Expr::One => true,
                            _ => {
                                return Err(ParseError::new(
                                    "Right-hand side of assignment 'xN :=' must be '0' or '1'"
                                        .to_string(),
                                    // Ideally, span of the rhs_expr. For now, operator span (or current_token's for it)
                                    // Let's use the span of the token that formed rhs_expr.
                                    // parse_primary consumes the token, so rhs_expr.span() would be ideal if Exp had spans.
                                    // For now, using the operator's span is a placeholder.
                                    // The original code used next_token_after_field.span, which is the operator.
                                    next_token_after_field.span,
                                ));
                            }
                        };
                        Ok(Expr::assign(self.arena, index, value_bool))
                    }
                    TokenKind::Eq => {
                        self.consume_token()?; // Consume '=='

                        // Peek for the value token (0 or 1)
                        let value_token_peek = self.peek_token().map_err(|e| e.clone())?;
                        if value_token_peek.kind == TokenKind::Eof {
                            return Err(ParseError::new(
                                format!(
                                    "Expected '0' or '1' after 'x{} ==' but found end of input",
                                    index
                                ),
                                value_token_peek.span, // Span of the EOF token
                            ));
                        }
                        // If not EOF, proceed to parse it (original logic)
                        let rhs_expr = self.parse_primary()?; // Use parse_primary for 0 or 1
                        let value_bool = match *rhs_expr {
                            Expr::Zero => false,
                            Expr::One => true,
                            _ => {
                                return Err(ParseError::new(
                                    "Right-hand side of test 'xN ==' must be '0' or '1'"
                                        .to_string(),
                                    // Using the operator's span as a placeholder, similar to Assign.
                                    next_token_after_field.span,
                                ));
                            }
                        };
                        Ok(Expr::test(self.arena, index, value_bool))
                    }
                    TokenKind::Tilde => {
                        // Pattern match: xN ~ pattern
                        self.consume_token()?; // Consume '~'

                        // Parse pattern
                        let pattern = self.parse_pattern(None)?;

                        // Convert to test based on pattern
                        match pattern {
                            Pattern::Exact(bits) if bits.len() == 1 => {
                                // Single bit pattern, convert to test
                                Ok(Expr::test(self.arena, index, bits[0]))
                            }
                            _ => Err(ParseError::new(
                                format!(
                                    "Field 'x{}' with '~' operator only supports patterns '0' or '1'",
                                    index
                                ),
                                next_token_after_field.span,
                            )),
                        }
                    }
                    _ => Err(ParseError::new(
                        format!(
                            "Expected ':=' or '~' after field 'x{}', found {}",
                            index,
                            token_kind_to_user_string(&next_token_after_field.kind)
                        ),
                        next_token_after_field.span,
                    )),
                }
            }
            // If expression
            TokenKind::If => {
                self.consume_token()?; // Consume 'if'
                let cond = self.parse_until()?; // Parse condition

                // Expect 'then'
                match self.consume_token()? {
                    Token {
                        kind: TokenKind::Then,
                        ..
                    } => {}
                    tok => {
                        return Err(ParseError::new(
                            format!(
                                "Expected 'then' after condition, but found {}",
                                token_kind_to_user_string(&tok.kind)
                            ),
                            tok.span,
                        ));
                    }
                }

                let then_expr = self.parse_until()?; // Parse then branch

                // Expect 'else'
                match self.consume_token()? {
                    Token {
                        kind: TokenKind::Else,
                        ..
                    } => {}
                    tok => {
                        return Err(ParseError::new(
                            format!(
                                "Expected 'else' after then branch, but found {}",
                                token_kind_to_user_string(&tok.kind)
                            ),
                            tok.span,
                        ));
                    }
                }

                let else_expr = self.parse_until()?; // Parse else branch

                Ok(Expr::if_then_else(self.arena, cond, then_expr, else_expr))
            }
            // Let expression
            TokenKind::Let => {
                self.consume_token()?; // Consume 'let'

                // Expect identifier
                let var_name = match self.consume_token()? {
                    Token {
                        kind: TokenKind::Ident(name),
                        ..
                    } => name,
                    tok => {
                        return Err(ParseError::new(
                            format!(
                                "Expected variable name after 'let', but found {}",
                                token_kind_to_user_string(&tok.kind)
                            ),
                            tok.span,
                        ));
                    }
                };

                // Expect '='
                match self.consume_token()? {
                    Token {
                        kind: TokenKind::Eq,
                        ..
                    } => {}
                    tok => {
                        return Err(ParseError::new(
                            format!(
                                "Expected '=' after variable name, but found {}",
                                token_kind_to_user_string(&tok.kind)
                            ),
                            tok.span,
                        ));
                    }
                }

                // Check if this is a bit range alias by looking for &x[...]
                if self.peek_kind()? == &TokenKind::And {
                    // This is a bit range alias: let alias = &x[start..end] in expr
                    self.consume_token()?; // Consume '&'

                    // Expect identifier (x or alias name)
                    let base_token = self.consume_token()?;
                    let base_name = match &base_token.kind {
                        TokenKind::Ident(name) => name,
                        _ => {
                            return Err(ParseError::new(
                                format!(
                                    "Expected identifier after '&' in bit range alias, but found {}",
                                    token_kind_to_user_string(&base_token.kind)
                                ),
                                base_token.span,
                            ));
                        }
                    };

                    // Expect '['
                    match self.consume_token()? {
                        Token {
                            kind: TokenKind::LBracket,
                            ..
                        } => {}
                        tok => {
                            return Err(ParseError::new(
                                format!(
                                    "Expected '[' after '&x' in bit range alias, but found {}",
                                    token_kind_to_user_string(&tok.kind)
                                ),
                                tok.span,
                            ));
                        }
                    }

                    // Parse start index
                    let start_token = self.consume_token()?;
                    let start = match start_token.kind {
                        TokenKind::Number(num_str) => num_str.parse::<u32>().map_err(|_| {
                            ParseError::new(
                                format!("Invalid start index in bit range alias: {}", num_str),
                                start_token.span,
                            )
                        })?,
                        _ => {
                            return Err(ParseError::new(
                                format!(
                                    "Expected number for start index, found {}",
                                    token_kind_to_user_string(&start_token.kind)
                                ),
                                start_token.span,
                            ));
                        }
                    };

                    // Expect '..'
                    match self.consume_token()? {
                        Token {
                            kind: TokenKind::DotDot,
                            ..
                        } => {}
                        tok => {
                            return Err(ParseError::new(
                                "Expected '..' in bit range alias".to_string(),
                                tok.span,
                            ));
                        }
                    }

                    // Parse end index
                    let end_token = self.consume_token()?;
                    let end = match end_token.kind {
                        TokenKind::Number(num_str) => num_str.parse::<u32>().map_err(|_| {
                            ParseError::new(
                                format!("Invalid end index in bit range alias: {}", num_str),
                                end_token.span,
                            )
                        })?,
                        _ => {
                            return Err(ParseError::new(
                                format!(
                                    "Expected number for end index, found {}",
                                    token_kind_to_user_string(&end_token.kind)
                                ),
                                end_token.span,
                            ));
                        }
                    };

                    // Expect ']'
                    match self.consume_token()? {
                        Token {
                            kind: TokenKind::RBracket,
                            ..
                        } => {}
                        tok => {
                            return Err(ParseError::new(
                                "Expected ']' after bit range alias".to_string(),
                                tok.span,
                            ));
                        }
                    }

                    // Expect 'in'
                    match self.consume_token()? {
                        Token {
                            kind: TokenKind::In,
                            ..
                        } => {}
                        tok => {
                            return Err(ParseError::new(
                                format!(
                                    "Expected 'in' after bit range alias, but found {}",
                                    token_kind_to_user_string(&tok.kind)
                                ),
                                tok.span,
                            ));
                        }
                    }

                    let body = self.parse_until()?; // Parse body

                    // For now, only support 'x' as the base
                    if *base_name != "x" {
                        return Err(ParseError::new(
                            format!(
                                "Sub-ranges of aliases not yet supported. Use &x[start..end] instead of &{}[start..end]",
                                base_name
                            ),
                            base_token.span,
                        ));
                    }

                    Ok(Expr::let_bit_range(self.arena, var_name, start, end, body))
                } else {
                    // Regular let binding
                    let def = self.parse_until()?; // Parse definition

                    // Expect 'in'
                    match self.consume_token()? {
                        Token {
                            kind: TokenKind::In,
                            ..
                        } => {}
                        tok => {
                            return Err(ParseError::new(
                                format!(
                                    "Expected 'in' after definition, but found {}",
                                    token_kind_to_user_string(&tok.kind)
                                ),
                                tok.span,
                            ));
                        }
                    }

                    let body = self.parse_until()?; // Parse body

                    Ok(Expr::let_in(self.arena, var_name, def, body))
                }
            }
            // Variable reference or bit range
            TokenKind::Ident(name) => {
                // Look ahead to see if this is a bit range
                if name == "x" {
                    // Try to peek at the next token after 'x'
                    // We need to temporarily consume 'x' to peek at what follows
                    let _x_token = self.consume_token()?; // Consume 'x'

                    if matches!(self.peek_kind(), Ok(&TokenKind::LBracket)) {
                        // This is a bit range expression x[start..end]
                        // 'x' is already consumed above
                        self.consume_token()?; // Consume '['

                        // Parse start index
                        let start_token = self.consume_token()?;
                        let start = match start_token.kind {
                            TokenKind::Number(num_str) => num_str.parse::<u32>().map_err(|_| {
                                ParseError::new(
                                    format!("Invalid start index in bit range: {}", num_str),
                                    start_token.span,
                                )
                            })?,
                            _ => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected number for bit range start, found {}",
                                        token_kind_to_user_string(&start_token.kind)
                                    ),
                                    start_token.span,
                                ));
                            }
                        };

                        // Expect '..'
                        match self.consume_token()? {
                            Token {
                                kind: TokenKind::DotDot,
                                ..
                            } => {}
                            tok => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected '..' in bit range, found {}",
                                        token_kind_to_user_string(&tok.kind)
                                    ),
                                    tok.span,
                                ));
                            }
                        }

                        // Parse end index
                        let end_token = self.consume_token()?;
                        let end = match end_token.kind {
                            TokenKind::Number(num_str) => num_str.parse::<u32>().map_err(|_| {
                                ParseError::new(
                                    format!("Invalid end index in bit range: {}", num_str),
                                    end_token.span,
                                )
                            })?,
                            _ => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected number for bit range end, found {}",
                                        token_kind_to_user_string(&end_token.kind)
                                    ),
                                    end_token.span,
                                ));
                            }
                        };

                        // Expect ']'
                        match self.consume_token()? {
                            Token {
                                kind: TokenKind::RBracket,
                                ..
                            } => {}
                            tok => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected ']' to close bit range, found {}",
                                        token_kind_to_user_string(&tok.kind)
                                    ),
                                    tok.span,
                                ));
                            }
                        }

                        // Check what follows: := or ==
                        let op_token = self.peek_token()?.clone();
                        match op_token.kind {
                            TokenKind::Assign => {
                                self.consume_token()?; // Consume ':='

                                // Parse the literal value
                                let value_token = self.consume_token()?;
                                let (value_str, literal_type) = match value_token.kind {
                                    TokenKind::Number(num_str) => (num_str, LiteralType::Decimal),
                                    TokenKind::BinaryLiteral(bin_str) => {
                                        (bin_str, LiteralType::Binary)
                                    }
                                    TokenKind::HexLiteral(hex_str) => {
                                        (hex_str, LiteralType::Hexadecimal)
                                    }
                                    TokenKind::IpLiteral(ip_str) => {
                                        (ip_str, LiteralType::IpAddress)
                                    }
                                    _ => {
                                        return Err(ParseError::new(
                                            format!(
                                                "Expected number after bit range assignment, found {}",
                                                token_kind_to_user_string(&value_token.kind)
                                            ),
                                            value_token.span,
                                        ));
                                    }
                                };

                                // Convert literal to bit vector
                                let bits = literal_to_bits(
                                    self.arena,
                                    value_str,
                                    literal_type,
                                    (end - start) as usize,
                                )?;
                                Ok(self.arena.alloc(Expr::BitRangeAssign(start, end, bits)))
                            }
                            TokenKind::Tilde => {
                                // Pattern match operator x[start..end] ~ pattern
                                self.consume_token()?; // Consume '~'

                                // Parse pattern with field width
                                let field_width = (end - start) as usize;
                                let pattern = self.parse_pattern(Some(field_width))?;
                                Ok(Expr::bit_range_match(self.arena, start, end, pattern))
                            }
                            _ => Err(ParseError::new(
                                format!(
                                    "Expected ':=' or '~' after bit range, found {}",
                                    token_kind_to_user_string(&op_token.kind)
                                ),
                                op_token.span,
                            )),
                        }
                    } else {
                        // Not a bit range, 'x' is already consumed, just return it as a variable
                        Ok(Expr::var(self.arena, name))
                    }
                } else {
                    // Check if this identifier is followed by := or ==
                    self.consume_token()?; // Consume the identifier

                    // Look ahead for assignment or test
                    match self.peek_kind()? {
                        TokenKind::Assign => {
                            // Variable assignment: var := value
                            self.consume_token()?; // Consume ':='

                            // Parse the value (should be a number or literal that evaluates to bits)
                            let value_token = self.consume_token()?;
                            let (value_str, literal_type) = match value_token.kind {
                                TokenKind::Number(num_str) => (num_str, LiteralType::Decimal),
                                TokenKind::BinaryLiteral(bin_str) => (bin_str, LiteralType::Binary),
                                TokenKind::HexLiteral(hex_str) => {
                                    (hex_str, LiteralType::Hexadecimal)
                                }
                                TokenKind::IpLiteral(ip_str) => (ip_str, LiteralType::IpAddress),
                                _ => {
                                    return Err(ParseError::new(
                                        format!(
                                            "Expected value after variable assignment, found {}",
                                            token_kind_to_user_string(&value_token.kind)
                                        ),
                                        value_token.span,
                                    ));
                                }
                            };

                            // For variable assignment, infer bit width from literal format
                            let inferred_bits = infer_literal_bit_width(value_str, literal_type)?;
                            let bits = literal_to_bits(
                                self.arena,
                                value_str,
                                literal_type,
                                inferred_bits,
                            )?;
                            Ok(self.arena.alloc(Expr::VarAssign(name, bits)))
                        }
                        TokenKind::Tilde => {
                            // Pattern match: var ~ pattern
                            self.consume_token()?; // Consume '~'

                            // Parse pattern (no field width for variables)
                            let pattern = self.parse_pattern(None)?;
                            Ok(Expr::var_match(self.arena, name, pattern))
                        }
                        _ => {
                            // Just a variable reference
                            Ok(Expr::var(self.arena, name))
                        }
                    }
                }
            }
            // These are simple primaries, handled by the simplified parse_primary
            TokenKind::Number(n) if n == "0" || n == "1" => self.parse_primary(),
            TokenKind::Top | TokenKind::Dup | TokenKind::LParen | TokenKind::End => {
                self.parse_primary()
            }
            // Any other token here is unexpected when trying to parse an atom or field expression
            _ => Err(ParseError::new(
                format!(
                    "Unexpected {} when expecting an expression",
                    token_kind_to_user_string(&current_token.kind)
                ),
                current_token.span,
            )),
        }
    }

    // Simplified parse_primary: handles only 0, 1, T, dup, and parenthesized expressions.
    // Field-related constructs are now handled by parse_atom_or_field_expression.
    fn parse_primary(&mut self) -> Result<Exp<'a>, ParseError> {
        let token = self.consume_token()?; // Consume the token for the primary
        match token.kind {
            TokenKind::Number("0") => Ok(Expr::zero(self.arena)),
            TokenKind::Number("1") => Ok(Expr::one(self.arena)),
            TokenKind::Top => Ok(Expr::top(self.arena)),
            TokenKind::Dup => Ok(Expr::dup(self.arena)),
            TokenKind::LParen => {
                let expr = self.parse_until()?;
                match self.consume_token()? {
                    // Expect RParen
                    Token {
                        kind: TokenKind::RParen,
                        ..
                    } => Ok(expr),
                    tok => Err(ParseError::new(
                        format!(
                            "Expected character ')' to close parenthesis, but found {}",
                            token_kind_to_user_string(&tok.kind)
                        ),
                        tok.span,
                    )),
                }
            }
            // Field is handled by parse_atom_or_field_expression.
            // Other tokens are invalid starts for a primary expression.
            TokenKind::Field(_)
            | TokenKind::Ident(_)
            | TokenKind::LtlX
            | TokenKind::LtlF
            | TokenKind::LtlG
            | TokenKind::LtlU
            | TokenKind::LtlR
            | TokenKind::Tilde
            | TokenKind::Not
            | TokenKind::Star
            | TokenKind::Semicolon
            | TokenKind::Plus
            | TokenKind::And
            | TokenKind::Xor
            | TokenKind::Minus
            | TokenKind::Assign
            | TokenKind::Eq
            | TokenKind::RParen
            | TokenKind::LBracket
            | TokenKind::RBracket
            | TokenKind::DotDot
            | TokenKind::Number(_)
            | TokenKind::BinaryLiteral(_)
            | TokenKind::HexLiteral(_)
            | TokenKind::IpLiteral(_)
            | TokenKind::If
            | TokenKind::Then
            | TokenKind::Else
            | TokenKind::Let
            | TokenKind::In
            | TokenKind::Lte
            | TokenKind::Hole
            | TokenKind::Def
            | TokenKind::Eof => {
                // Removed End from here due to unreachable pattern, it's handled below.
                Err(ParseError::new(
                    format!(
                        "Unexpected {} when expecting a primary expression (like '0', '1', 'T', 'dup', or '(')",
                        token_kind_to_user_string(&token.kind)
                    ),
                    token.span,
                ))
            }
            TokenKind::End => Ok(Expr::end(self.arena)), // Moved here to be last, resolves unreachable_patterns for End.
        }
    }

    /// Parse a pattern for pattern matching expressions
    fn parse_pattern(&mut self, field_width: Option<usize>) -> Result<Pattern<'a>, ParseError> {
        let token = self.consume_token()?;

        match token.kind {
            TokenKind::IpLiteral(ip_str) => {
                // Check if this is CIDR notation (contains /)
                if let Some(slash_pos) = ip_str.find('/') {
                    let (addr_str, prefix_str) = ip_str.split_at(slash_pos);
                    let prefix_str = &prefix_str[1..]; // Skip the '/'

                    // Parse IP address
                    let addr_bits = ip_to_bits(self.arena, addr_str)?;

                    // Parse prefix length
                    let prefix_len = prefix_str.parse::<usize>().map_err(|_| {
                        ParseError::new(
                            format!("Invalid CIDR prefix length: {}", prefix_str),
                            token.span,
                        )
                    })?;

                    if prefix_len > 32 {
                        return Err(ParseError::new(
                            format!("CIDR prefix length {} exceeds maximum of 32", prefix_len),
                            token.span,
                        ));
                    }

                    Ok(Pattern::Cidr {
                        address: addr_bits,
                        prefix_len,
                    })
                } else if ip_str.contains('-') {
                    // IP range: start-end
                    let parts: bumpalo::collections::Vec<_> =
                        bumpalo::collections::Vec::from_iter_in(ip_str.split('-'), self.arena);
                    if parts.len() != 2 {
                        return Err(ParseError::new(
                            format!("Invalid range format: {}", ip_str),
                            token.span,
                        ));
                    }

                    let start_bits = ip_to_bits(self.arena, parts[0])?;
                    let end_bits = ip_to_bits(self.arena, parts[1])?;

                    Ok(Pattern::IpRange {
                        start: start_bits,
                        end: end_bits,
                    })
                } else {
                    // Check if next token is "mask" for wildcard syntax
                    if matches!(self.peek_kind(), Ok(TokenKind::Ident(s)) if *s == "mask") {
                        self.consume_token()?; // Consume "mask"

                        // Parse mask value
                        let mask_token = self.consume_token()?;
                        let mask_bits = match mask_token.kind {
                            TokenKind::IpLiteral(mask_str) => ip_to_bits(self.arena, mask_str)?,
                            TokenKind::Number(num_str) => {
                                literal_to_bits(self.arena, num_str, LiteralType::Decimal, 32)?
                            }
                            TokenKind::HexLiteral(hex_str) => {
                                literal_to_bits(self.arena, hex_str, LiteralType::Hexadecimal, 32)?
                            }
                            _ => {
                                return Err(ParseError::new(
                                    format!(
                                        "Expected mask value, found {}",
                                        token_kind_to_user_string(&mask_token.kind)
                                    ),
                                    mask_token.span,
                                ));
                            }
                        };

                        let addr_bits = ip_to_bits(self.arena, ip_str)?;
                        Ok(Pattern::Wildcard {
                            address: addr_bits,
                            mask: mask_bits,
                        })
                    } else {
                        // Just an exact IP match - use ip_to_bits for consistent MSB-first order
                        let bits = ip_to_bits(self.arena, ip_str)?;
                        Ok(Pattern::Exact(bits))
                    }
                }
            }
            TokenKind::Number(num_str) => {
                // Check for CIDR notation with regular number
                if matches!(self.peek_kind(), Ok(&TokenKind::Minus)) {
                    // This might be a range
                    self.consume_token()?; // Consume '-'

                    let end_token = self.consume_token()?;
                    let end_str = match end_token.kind {
                        TokenKind::Number(s) => s,
                        TokenKind::IpLiteral(s) => s,
                        _ => {
                            return Err(ParseError::new(
                                format!(
                                    "Expected number after '-' in range, found {}",
                                    token_kind_to_user_string(&end_token.kind)
                                ),
                                end_token.span,
                            ));
                        }
                    };

                    // Use field width if provided, otherwise default to minimal width
                    let start_val = num_str.parse::<u128>().map_err(|_| {
                        ParseError::new(format!("Invalid number: {}", num_str), token.span)
                    })?;
                    let end_val = if end_str.contains('.') {
                        // This is an IP address end point, parse it as IP
                        let end_bits = ip_to_bits(self.arena, end_str)?;
                        bits_to_u128(end_bits).map_err(|_| {
                            ParseError::new(
                                "IP address too large for range".to_string(),
                                end_token.span,
                            )
                        })?
                    } else {
                        end_str.parse::<u128>().map_err(|_| {
                            ParseError::new(format!("Invalid number: {}", end_str), end_token.span)
                        })?
                    };

                    // Determine bit width needed for the range
                    let max_val = std::cmp::max(start_val, end_val);
                    let min_width = if max_val == 0 {
                        1
                    } else {
                        (128 - max_val.leading_zeros()) as usize
                    };

                    // Use field width if provided and sufficient, otherwise use minimal width
                    let width = match field_width {
                        Some(fw) if fw >= min_width => fw,
                        _ => min_width,
                    };

                    let start_bits =
                        literal_to_bits(self.arena, num_str, LiteralType::Decimal, width)?;
                    let end_bits = if end_str.contains('.') {
                        ip_to_bits(self.arena, end_str)?
                    } else {
                        literal_to_bits(self.arena, end_str, LiteralType::Decimal, width)?
                    };

                    Ok(Pattern::IpRange {
                        start: start_bits,
                        end: end_bits,
                    })
                } else {
                    // Just an exact match
                    let val = num_str.parse::<u128>().map_err(|_| {
                        ParseError::new(format!("Invalid number: {}", num_str), token.span)
                    })?;

                    // Determine bit width
                    let min_width = if val == 0 {
                        1
                    } else {
                        (128 - val.leading_zeros()) as usize
                    };

                    let width = match field_width {
                        Some(fw) if fw >= min_width => fw,
                        _ => min_width,
                    };

                    let bits = literal_to_bits(self.arena, num_str, LiteralType::Decimal, width)?;
                    Ok(Pattern::Exact(bits))
                }
            }
            TokenKind::BinaryLiteral(bin_str) => {
                let inferred_width = infer_literal_bit_width(bin_str, LiteralType::Binary)?;
                let bits =
                    literal_to_bits(self.arena, bin_str, LiteralType::Binary, inferred_width)?;
                Ok(Pattern::Exact(bits))
            }
            TokenKind::HexLiteral(hex_str) => {
                let inferred_width = infer_literal_bit_width(hex_str, LiteralType::Hexadecimal)?;
                let bits = literal_to_bits(
                    self.arena,
                    hex_str,
                    LiteralType::Hexadecimal,
                    inferred_width,
                )?;
                Ok(Pattern::Exact(bits))
            }
            _ => Err(ParseError::new(
                format!(
                    "Expected pattern (IP address, number, or literal), found {}",
                    token_kind_to_user_string(&token.kind)
                ),
                token.span,
            )),
        }
    }
}

/// Convert bit vector to u128
fn bits_to_u128(bits: &[bool]) -> Result<u128, ParseError> {
    if bits.len() > 128 {
        return Err(ParseError::new(
            "Bit vector too large (max 128 bits)".to_string(),
            Default::default(),
        ));
    }

    let mut val = 0u128;
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            val |= 1u128 << (bits.len() - 1 - i);
        }
    }
    Ok(val)
}

fn ip_to_number(ip_str: &str) -> Result<u32, ParseError> {
    let parts = ip_str.split('.');
    if parts.clone().count() != 4 {
        return Err(ParseError::new(
            format!("Invalid IP address format: {}", ip_str),
            Default::default(),
        ));
    }
    let mut ip_num = 0u32;
    for (i, part) in parts.enumerate() {
        let octet = part.parse::<u8>().map_err(|_| {
            ParseError::new(format!("Invalid IP octet: {}", part), Default::default())
        })?;
        ip_num |= (octet as u32) << (8 * (3 - i));
    }
    Ok(ip_num)
}

fn ip_to_bits<'a>(arena: &'a Bump, ip_str: &str) -> Result<&'a [bool], ParseError> {
    let num = ip_to_number(ip_str)?;
    Ok(arena.alloc_slice_fill_iter((0..32).map(|i| (num >> i) & 1 == 1)))
}

#[derive(Debug, Clone, Copy)]
enum LiteralType {
    Decimal,
    Binary,
    Hexadecimal,
    IpAddress,
}

// Helper function to convert various literal formats to a bit vector
// Infer bit width based on literal format
fn infer_literal_bit_width(
    literal_str: &str,
    literal_type: LiteralType,
) -> Result<usize, ParseError> {
    match literal_type {
        LiteralType::Binary => {
            // Binary literal: bit width = number of digits
            Ok(literal_str.len())
        }
        LiteralType::Hexadecimal => {
            // Hex literal: bit width = 4 * number of digits
            Ok(literal_str.len() * 4)
        }
        LiteralType::IpAddress => {
            // IP address: always 32 bits
            Ok(32)
        }
        LiteralType::Decimal => {
            // Decimal: use minimal bit width
            let num = literal_str.parse::<u128>().map_err(|_| {
                ParseError::new(
                    format!("Invalid decimal number: {}", literal_str),
                    Default::default(),
                )
            })?;

            // Find minimal bit width needed
            if num == 0 {
                Ok(1)
            } else {
                // Calculate bits needed: floor(log2(num)) + 1
                Ok((128 - num.leading_zeros()) as usize)
            }
        }
    }
}

fn literal_to_bits<'a>(
    arena: &'a Bump,
    literal_str: &str,
    literal_type: LiteralType,
    expected_bits: usize,
) -> Result<&'a [bool], ParseError> {
    let num = match literal_type {
        LiteralType::Decimal => literal_str.parse::<u128>().map_err(|_| {
            ParseError::new(
                format!("Invalid decimal number: {}", literal_str),
                Default::default(),
            )
        })?,
        LiteralType::Binary => u128::from_str_radix(literal_str, 2).map_err(|_| {
            ParseError::new(
                format!("Invalid binary literal: 0b{}", literal_str),
                Default::default(),
            )
        })?,
        LiteralType::Hexadecimal => u128::from_str_radix(literal_str, 16).map_err(|_| {
            ParseError::new(
                format!("Invalid hexadecimal literal: 0x{}", literal_str),
                Default::default(),
            )
        })?,
        LiteralType::IpAddress => ip_to_number(literal_str)? as u128,
    };

    // Check if the number fits in the expected bits
    if expected_bits < 128 && num >= (1u128 << expected_bits) {
        return Err(ParseError::new(
            format!("Number {} requires more than {} bits", num, expected_bits),
            Default::default(),
        ));
    }

    Ok(arena.alloc_slice_fill_iter((0..expected_bits).map(|i| i < 128 && (num >> i) & 1 == 1)))
}

// Helper function to convert a number string to a bit vector (for backward compatibility)
fn number_to_bits<'a>(
    arena: &'a Bump,
    num_str: &str,
    expected_bits: usize,
) -> Result<&'a [bool], ParseError> {
    literal_to_bits(arena, num_str, LiteralType::Decimal, expected_bits)
}

/// Parse into caller-owned arena storage, borrowing names from `input`.
///
/// The shared lifetime is the intersection of the input and arena lifetimes;
/// neither must have the same original scope as the other. The returned slice
/// is empty for whitespace/comments, or contains the single parsed expression.
pub fn parse_expressions<'a>(
    arena: &'a Bump,
    input: &'a str,
) -> Result<&'a [Exp<'a>], ParseErrorDetails> {
    let lexer = Lexer::new(input);
    let mut parser = Parser::new(lexer, arena);
    let mut expressions = bumpalo::collections::Vec::new_in(arena);

    // This parses at most one top-level expression. The slice is empty for
    // whitespace/comments; otherwise it contains one expression. Historically
    // callers expected a collection, and the trailing peek-for-Eof remains: every caller reads only
    // `expressions[0]`, and any token after the first expression is reported
    // as "Expected operator". The empty result is the "input was just
    // whitespace/comments" case.
    match parser.peek_kind() {
        Ok(&TokenKind::Eof) => return Ok(expressions.into_bump_slice()),
        Err(ref pe)
            if pe.message.contains("Peeked beyond EOF")
                || pe.message.contains("Unexpected end of token stream") =>
        {
            return Ok(expressions.into_bump_slice());
        }
        Err(pe) => return Err(convert_parse_error(pe.clone(), input)),
        _ => {}
    }

    match parser.parse_single_expression() {
        Ok(expr) => expressions.push(expr),
        Err(pe) => return Err(convert_parse_error(pe, input)),
    }

    match parser.peek_kind() {
        Ok(&TokenKind::Eof) => Ok(expressions.into_bump_slice()),
        Err(ref pe)
            if pe.message.contains("Peeked beyond EOF")
                || pe.message.contains("Unexpected end of token stream") =>
        {
            Ok(expressions.into_bump_slice())
        }
        Ok(kind_from_first_peek) => {
            let owned_kind = kind_from_first_peek.clone();
            let span_for_error = match parser.peek_token() {
                Ok(token) => token.span,
                Err(_) => Default::default(),
            };
            let err = ParseError::new(
                if matches!(owned_kind, TokenKind::Eq) {
                    "The '==' operator is only supported for simple field tests like 'x0 == 1'. For other comparisons, use the '~' operator instead (e.g., 'port ~ 1024')".to_string()
                } else {
                    format!(
                        "Expected operator, but found {}",
                        token_kind_to_user_string(&owned_kind)
                    )
                },
                span_for_error,
            );
            Err(convert_parse_error(err, input))
        }
        Err(pe) => Err(convert_parse_error(pe.clone(), input)),
    }
}

// Helper to convert internal ParseError to the lib's ParseErrorDetails
fn convert_parse_error(pe: ParseError, _input: &str) -> ParseErrorDetails {
    // In a more advanced scenario, `input` could be used with `pe.span.offset`
    // to find line/column if `pe.span` only had offsets. But our `pe.span` has line/col.
    ParseErrorDetails {
        message: pe.message,
        span: Some(ErrorSpan {
            // Using ErrorSpan directly, assuming it's in scope via `use crate::ErrorSpan`
            start_line: pe.span.start.line,
            start_column: pe.span.start.column,
            end_line: pe.span.end.line,
            // End column for Monaco is typically exclusive for single char, inclusive for multi-char.
            // Our `end.column` is the column *after* the last char of the token.
            // For a single char token at col 5, start.col=5, end.col=6.
            // Monaco marker: startColumn, endColumn. If endColumn is startColumn+1, it's one char.
            // Let's make end_column inclusive for Monaco.
            // If token is 1 char, start=5, end=6. Monaco: start=5, end=6. Correct.
            // If token "dup" starts col 5: d(5) u(6) p(7). start.col=5, end.col=8.
            // Monaco: start=5, end=8. Correct (highlights d,u,p).
            end_column: pe.span.end.column,
        }),
    }
}

// Helper to convert TokenKind to a user-friendly string representation
fn token_kind_to_user_string(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Top => "keyword 'T'".to_string(),
        TokenKind::Assign => "operator ':='".to_string(),
        TokenKind::Eq => "operator '=='".to_string(),
        TokenKind::Lte => "operator '<='".to_string(),
        TokenKind::Plus => "operator '+'".to_string(),
        TokenKind::And => "operator '&'".to_string(),
        TokenKind::Xor => "operator '^'".to_string(),
        TokenKind::Minus => "operator '-'".to_string(),
        TokenKind::Tilde => "operator '~'".to_string(),
        TokenKind::Not => "operator '!'".to_string(),
        TokenKind::Semicolon => "operator ';'".to_string(),
        TokenKind::Star => "operator '*'".to_string(),
        TokenKind::Dup => "keyword 'dup'".to_string(),
        TokenKind::LtlX => "LTL operator 'X'".to_string(),
        TokenKind::LtlU => "LTL operator 'U'".to_string(),
        TokenKind::LtlF => "LTL operator 'F'".to_string(),
        TokenKind::LtlG => "LTL operator 'G'".to_string(),
        TokenKind::LtlR => "LTL operator 'R'".to_string(),
        TokenKind::LParen => "character '('".to_string(),
        TokenKind::RParen => "character ')'".to_string(),
        TokenKind::LBracket => "character '['".to_string(),
        TokenKind::RBracket => "character ']'".to_string(),
        TokenKind::DotDot => "operator '..'".to_string(),
        TokenKind::Field(u) => format!("field 'x{}'", u),
        TokenKind::Number(n) => format!("number '{}'", n),
        TokenKind::BinaryLiteral(b) => format!("binary literal '0b{}'", b),
        TokenKind::HexLiteral(h) => format!("hexadecimal literal '0x{}'", h),
        TokenKind::IpLiteral(ip) => format!("IP literal '{}'", ip),
        TokenKind::Ident(s) => format!("identifier '{}'", s),
        TokenKind::If => "keyword 'if'".to_string(),
        TokenKind::Then => "keyword 'then'".to_string(),
        TokenKind::Else => "keyword 'else'".to_string(),
        TokenKind::Let => "keyword 'let'".to_string(),
        TokenKind::In => "keyword 'in'".to_string(),
        TokenKind::End => "keyword 'end'".to_string(),
        TokenKind::Hole => "keyword 'hole'".to_string(),
        TokenKind::Def => "keyword 'def'".to_string(),
        TokenKind::Eof => "end of input".to_string(),
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::{Exp, parse_expressions}; // Removed Lexer, Parser, TokenKind
    use crate::expr::{Expr, Pattern};

    #[test]
    fn identifiers_borrow_the_original_utf8_source() {
        let source = String::from("let café = 1 in café");
        let arena = bumpalo::Bump::new();
        let parsed = parse_expressions(&arena, &source).unwrap();
        let Expr::Let(name, _, body) = parsed[0] else {
            panic!("expected let")
        };
        let Expr::Var(reference) = body else {
            panic!("expected variable")
        };
        for (name, offset) in [
            (*name, source.find("café").unwrap()),
            (*reference, source.rfind("café").unwrap()),
        ] {
            assert_eq!(name, "café");
            assert_eq!(name.as_ptr(), source[offset..].as_ptr());
        }
    }

    #[test]
    fn syntax_has_no_owned_heap_storage_or_recursive_drop() {
        use crate::holes::parser::{Program, Stmt};
        assert!(!std::mem::needs_drop::<Expr<'_>>());
        assert!(!std::mem::needs_drop::<Pattern<'_>>());
        assert!(!std::mem::needs_drop::<Program<'_>>());
        assert!(!std::mem::needs_drop::<Stmt<'_>>());
        let arena = bumpalo::Bump::new();
        let mut expr = Expr::one(&arena);
        for _ in 0..100_000 {
            expr = Expr::union(&arena, expr, Expr::zero(&arena));
        }
        assert!(matches!(expr, Expr::Union(..)));
        drop(arena); // A boxed tree this deep would need recursive destruction.
    }

    #[test]
    fn arena_bit_literals_support_full_width_and_zero_padding() {
        let arena = bumpalo::Bump::new();
        let parsed =
            parse_expressions(&arena, "x[0..128] := 0xffffffffffffffffffffffffffffffff").unwrap();
        let Expr::BitRangeAssign(0, 128, bits) = parsed[0] else {
            panic!("expected assignment")
        };
        assert_eq!(bits.len(), 128);
        assert!(bits.iter().all(|bit| *bit));
        let parsed = parse_expressions(&arena, "x[0..129] := 1").unwrap();
        let Expr::BitRangeAssign(0, 129, bits) = parsed[0] else {
            panic!("expected assignment")
        };
        assert!(bits[0]);
        assert!(bits[1..].iter().all(|bit| !bit));
    }

    // Helper to parse all expressions from a string and get a Vec<Exp>
    // Maps error to String for test assertion convenience.
    fn parse_all<'a>(arena: &'a bumpalo::Bump, s: &'a str) -> Result<Vec<Exp<'a>>, String> {
        match parse_expressions(arena, s) {
            Ok(exprs) => Ok(exprs.to_vec()),
            Err(details) => Err(details.message), // Using the message from ParseErrorDetails
        }
    }

    // Helper to parse a single expression expected from a string.
    // Fails if zero or more than one expression is parsed.
    fn parse_single_unwrap<'a>(arena: &'a bumpalo::Bump, s: &'a str) -> Exp<'a> {
        match parse_expressions(arena, s) {
            Ok(exprs) => {
                if exprs.len() == 1 {
                    exprs[0]
                } else {
                    panic!(
                        "Expected single expression, found {} for input: {}",
                        exprs.len(),
                        s
                    );
                }
            }
            Err(details) => panic!("Parse error for input '{}': {}", s, details.message),
        }
    }

    #[test]
    fn test_simple_literals() {
        let arena = &bumpalo::Bump::new();
        assert_eq!(parse_single_unwrap(arena, "0"), Expr::zero(arena));
        assert_eq!(parse_single_unwrap(arena, "1"), Expr::one(arena));
        assert_eq!(parse_single_unwrap(arena, "T"), Expr::top(arena));
        assert_eq!(parse_single_unwrap(arena, "dup"), Expr::dup(arena));
    }

    #[test]
    fn test_field_operations() {
        let arena = &bumpalo::Bump::new();
        assert_eq!(
            parse_single_unwrap(arena, "x1 := 0"),
            Expr::assign(arena, 1, false)
        );
        assert_eq!(
            parse_single_unwrap(arena, "x2 == 1"),
            Expr::test(arena, 2, true)
        );
    }

    #[test]
    fn test_sequence() {
        let arena = &bumpalo::Bump::new();
        assert_eq!(
            parse_single_unwrap(arena, "x0==1 ; x1==0"),
            Expr::sequence(
                arena,
                Expr::test(arena, 0, true),
                Expr::test(arena, 1, false)
            )
        );
    }

    #[test]
    fn test_union() {
        let arena = &bumpalo::Bump::new();
        assert_eq!(
            parse_single_unwrap(arena, "0 + 1"),
            Expr::union(arena, Expr::zero(arena), Expr::one(arena))
        );
    }

    #[test]
    fn test_parentheses() {
        let arena = &bumpalo::Bump::new();
        assert_eq!(
            parse_single_unwrap(arena, "(0+1)"),
            Expr::union(arena, Expr::zero(arena), Expr::one(arena))
        );
        assert_eq!(
            parse_single_unwrap(arena, "~(0)"),
            Expr::complement(arena, Expr::zero(arena))
        );
    }

    #[test]
    fn test_star() {
        let arena = &bumpalo::Bump::new();
        assert_eq!(
            parse_single_unwrap(arena, "0*"),
            Expr::star(arena, Expr::zero(arena))
        );
        assert_eq!(
            parse_single_unwrap(arena, "(x1==0)*"),
            Expr::star(arena, Expr::test(arena, 1, false))
        );
    }

    #[test]
    fn test_ltl_operators() {
        let arena = &bumpalo::Bump::new();
        assert_eq!(
            parse_single_unwrap(arena, "X 0"),
            Expr::ltl_next(arena, Expr::zero(arena))
        );
        // F e = T U e
        assert_eq!(
            parse_single_unwrap(arena, "F 0"),
            Expr::ltl_until(arena, Expr::top(arena), Expr::zero(arena))
        );
        // G e = !F!e = !(T U !e)
        assert_eq!(
            parse_single_unwrap(arena, "G 0"),
            Expr::ltl_globally(arena, Expr::zero(arena))
        );
        assert_eq!(
            parse_single_unwrap(arena, "0 U 1"),
            Expr::ltl_until(arena, Expr::zero(arena), Expr::one(arena))
        );
        // e1 R e2 = !( !e1 U !e2 )
        assert_eq!(
            parse_single_unwrap(arena, "0 R 1"),
            Expr::complement(
                arena,
                Expr::ltl_until(
                    arena,
                    Expr::complement(arena, Expr::zero(arena)),
                    Expr::complement(arena, Expr::one(arena))
                )
            )
        );
    }

    #[test]
    fn test_empty_input() {
        let arena = &bumpalo::Bump::new();
        assert!(parse_all(arena, "").unwrap().is_empty());
        assert!(parse_all(arena, "   ").unwrap().is_empty()); // Whitespace only
    }

    #[test]
    fn test_comments_only() {
        let arena = &bumpalo::Bump::new();
        assert!(parse_all(arena, "// this is a comment").unwrap().is_empty());
        assert!(
            parse_all(arena, "// comment 1\n// comment 2")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_expression_then_eof() {
        let arena = &bumpalo::Bump::new();
        assert_eq!(
            parse_single_unwrap(arena, "x7 == 0"),
            Expr::test(arena, 7, false)
        );
    }

    #[test]
    fn test_invalid_token() {
        let arena = &bumpalo::Bump::new();
        assert!(parse_all(arena, "$").is_err());
        assert!(parse_all(arena, "0 $").is_err());
    }

    #[test]
    fn test_incomplete_expressions() {
        let arena = &bumpalo::Bump::new();
        assert!(parse_all(arena, "x1 :=").is_err(), "Incomplete assignment");
        assert!(parse_all(arena, "x1 ==").is_err(), "Incomplete test");
        assert!(parse_all(arena, "0 +").is_err(), "Incomplete addition");
        assert!(parse_all(arena, "(").is_err(), "Unclosed parenthesis");
        assert!(parse_all(arena, "0 U").is_err(), "Incomplete Until");
    }

    #[test]
    fn test_unexpected_token_after_expr() {
        let arena = &bumpalo::Bump::new();
        // parse_expressions expects 'end' or EOF after an expression if there are multiple.
        // If single expression, it should be fine.
        assert!(
            parse_all(arena, "0 1").is_err(),
            "Expected 'end' or EOF after expr, not another expr"
        );
    }

    #[test]
    fn test_complex_expression() {
        let arena = &bumpalo::Bump::new();
        let expr_str = "(x1:=0 ; (T* & ~(x2==1))) + F (x3==0 U x4==1)";
        assert!(parse_all(arena, expr_str).is_ok());
        // More detailed check if needed by comparing the resulting Exp structure.
    }

    #[test]
    fn test_syntax_errors() {
        let arena = &bumpalo::Bump::new();
        let test_cases = vec![
            "x := 0",
            "x1 :=",
            "x1 : 0",
            "x1 = 0",
            "x1 == ",
            "0 +",
            "(0 + 1",
            "0 + 1)",
            "~ ",
            "F ",
            "X ",
            "0 U ",
            "0 R ",
            "foo",
            "x1 := T",
            "x1 == T",
            "0 1",
            "*0",
            "0;*1",
            ":",
            "==",
            "x",
            "e",
            "en",
            "du",
            "0 // comment", // This one should pass
            "0 // comment \n 1",
        ];

        println!("--- Checking Syntax Error Messages ---");
        for (i, input) in test_cases.iter().enumerate() {
            match parse_all(arena, input) {
                Ok(exprs) => {
                    if input.trim() == "0 // comment" {
                        // This one is valid
                        if exprs.len() == 1 && exprs[0] == Expr::zero(arena) {
                            println!(
                                "{}. Input: {:<15} -> PASSED (Correctly parsed)",
                                i + 1,
                                input
                            );
                        } else {
                            println!(
                                "{}. Input: {:<15} -> UNEXPECTED SUCCESS (but was expected to pass): {:?}",
                                i + 1,
                                input,
                                exprs
                            );
                        }
                    } else {
                        println!(
                            "{}. Input: {:<15} -> UNEXPECTED SUCCESS: {:?}",
                            i + 1,
                            input,
                            exprs
                        );
                    }
                }
                Err(e) => {
                    println!("{}. Input: {:<15} -> Error: \"{}\"", i + 1, input, e);
                }
            }
        }
        println!("--- Finished Checking Syntax Error Messages ---");
    }

    // ===== OPERATOR PRECEDENCE TESTS =====

    #[test]
    fn test_precedence_postfix_vs_prefix() {
        let arena = &bumpalo::Bump::new();
        // Postfix should bind tighter than prefix
        // !a* should be !(a*), not (!a)*
        assert_eq!(
            parse_single_unwrap(arena, "~0*"),
            parse_single_unwrap(arena, "~(0*)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "X 1*"),
            parse_single_unwrap(arena, "X (1*)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "F T*"),
            parse_single_unwrap(arena, "F (T*)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "G dup*"),
            parse_single_unwrap(arena, "G (dup*)")
        );
    }

    #[test]
    fn test_precedence_prefix_vs_infix() {
        let arena = &bumpalo::Bump::new();
        // Prefix should bind tighter than infix
        // !a + b should be (!a) + b, not !(a + b)
        assert_eq!(
            parse_single_unwrap(arena, "~0 + 1"),
            parse_single_unwrap(arena, "(~0) + 1")
        );
        assert_eq!(
            parse_single_unwrap(arena, "X 0 & 1"),
            parse_single_unwrap(arena, "(X 0) & 1")
        );
        assert_eq!(
            parse_single_unwrap(arena, "F 0 ; 1"),
            parse_single_unwrap(arena, "(F 0) ; 1")
        );
        assert_eq!(
            parse_single_unwrap(arena, "G 0 U 1"),
            parse_single_unwrap(arena, "(G 0) U 1")
        );
    }

    #[test]
    fn test_precedence_infix_vs_postfix() {
        let arena = &bumpalo::Bump::new();
        // Postfix should bind tighter than infix
        // a + b* should be a + (b*), not (a + b)*
        assert_eq!(
            parse_single_unwrap(arena, "0 + 1*"),
            parse_single_unwrap(arena, "0 + (1*)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "0 & T*"),
            parse_single_unwrap(arena, "0 & (T*)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "0 ; dup*"),
            parse_single_unwrap(arena, "0 ; (dup*)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "0 U 1*"),
            parse_single_unwrap(arena, "0 U (1*)")
        );
    }

    #[test]
    fn test_precedence_intersection_vs_sequence() {
        let arena = &bumpalo::Bump::new();
        // Intersection should bind tighter than sequence
        // a & b ; c should be (a & b) ; c
        assert_eq!(
            parse_single_unwrap(arena, "0 & 1 ; T"),
            parse_single_unwrap(arena, "(0 & 1) ; T")
        );
        // a ; b & c should be a ; (b & c)
        assert_eq!(
            parse_single_unwrap(arena, "0 ; 1 & T"),
            parse_single_unwrap(arena, "0 ; (1 & T)")
        );
    }

    #[test]
    fn test_precedence_sequence_vs_additive() {
        let arena = &bumpalo::Bump::new();
        // Sequence should bind tighter than additive (this was the original issue)
        // a ; b + c should be (a ; b) + c
        assert_eq!(
            parse_single_unwrap(arena, "0 ; 1 + T"),
            parse_single_unwrap(arena, "(0 ; 1) + T")
        );
        // a + b ; c should be a + (b ; c)
        assert_eq!(
            parse_single_unwrap(arena, "0 + 1 ; T"),
            parse_single_unwrap(arena, "0 + (1 ; T)")
        );

        // Same for other additive operators
        assert_eq!(
            parse_single_unwrap(arena, "0 ; 1 ^ T"),
            parse_single_unwrap(arena, "(0 ; 1) ^ T")
        );
        assert_eq!(
            parse_single_unwrap(arena, "0 ; 1 - T"),
            parse_single_unwrap(arena, "(0 ; 1) - T")
        );
    }

    #[test]
    fn test_precedence_additive_vs_until() {
        let arena = &bumpalo::Bump::new();
        // Additive should bind tighter than until/release
        // a + b U c should be (a + b) U c
        assert_eq!(
            parse_single_unwrap(arena, "0 + 1 U T"),
            parse_single_unwrap(arena, "(0 + 1) U T")
        );
        // a U b + c should be a U (b + c)
        assert_eq!(
            parse_single_unwrap(arena, "0 U 1 + T"),
            parse_single_unwrap(arena, "0 U (1 + T)")
        );

        // Same for release
        assert_eq!(
            parse_single_unwrap(arena, "0 + 1 R T"),
            parse_single_unwrap(arena, "(0 + 1) R T")
        );
    }

    #[test]
    fn test_associativity_left_associative() {
        let arena = &bumpalo::Bump::new();
        // Left-associative operators: &, ;, +, ^, -

        // Intersection
        assert_eq!(
            parse_single_unwrap(arena, "0 & 1 & T"),
            parse_single_unwrap(arena, "(0 & 1) & T")
        );

        // Sequence
        assert_eq!(
            parse_single_unwrap(arena, "0 ; 1 ; T"),
            parse_single_unwrap(arena, "(0 ; 1) ; T")
        );

        // Union
        assert_eq!(
            parse_single_unwrap(arena, "0 + 1 + T"),
            parse_single_unwrap(arena, "(0 + 1) + T")
        );

        // XOR
        assert_eq!(
            parse_single_unwrap(arena, "0 ^ 1 ^ T"),
            parse_single_unwrap(arena, "(0 ^ 1) ^ T")
        );

        // Difference
        assert_eq!(
            parse_single_unwrap(arena, "0 - 1 - T"),
            parse_single_unwrap(arena, "(0 - 1) - T")
        );
    }

    #[test]
    fn test_associativity_right_associative() {
        let arena = &bumpalo::Bump::new();
        // Right-associative operators: U, R, prefix operators

        // Until
        assert_eq!(
            parse_single_unwrap(arena, "0 U 1 U T"),
            parse_single_unwrap(arena, "0 U (1 U T)")
        );

        // Release
        assert_eq!(
            parse_single_unwrap(arena, "0 R 1 R T"),
            parse_single_unwrap(arena, "0 R (1 R T)")
        );

        // Prefix operators
        assert_eq!(
            parse_single_unwrap(arena, "~~0"),
            parse_single_unwrap(arena, "~(~0)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "~X 0"),
            parse_single_unwrap(arena, "~(X 0)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "X F 0"),
            parse_single_unwrap(arena, "X (F 0)")
        );
    }

    #[test]
    fn test_associativity_postfix() {
        let arena = &bumpalo::Bump::new();
        // Multiple stars should be left-associative: a** = (a*)*
        assert_eq!(
            parse_single_unwrap(arena, "0**"),
            parse_single_unwrap(arena, "(0*)*")
        );
        assert_eq!(
            parse_single_unwrap(arena, "T***"),
            parse_single_unwrap(arena, "((T*)*)*")
        );
    }

    #[test]
    fn test_mixed_precedence_complex() {
        let arena = &bumpalo::Bump::new();
        // Complex mixed precedence tests

        // !a* + b ; c & d U e should be ((!((a)*)) + (b ; (c & d))) U e
        assert_eq!(
            parse_single_unwrap(arena, "~0* + 1 ; T & dup U x1==0"),
            parse_single_unwrap(arena, "(~(0*) + (1 ; (T & dup))) U x1==0")
        );

        // a & !b* ; c + d should be ((a & (!(b*))) ; c) + d
        // Precedence: postfix (*), prefix (!), intersection (&), sequence (;), additive (+)
        assert_eq!(
            parse_single_unwrap(arena, "0 & ~1* ; T + dup"),
            parse_single_unwrap(arena, "((0 & ~(1*)) ; T) + dup")
        );
    }

    #[test]
    fn test_parentheses_override_precedence() {
        let arena = &bumpalo::Bump::new();
        // Parentheses should override precedence - test meaningful differences

        // Show that parentheses change the default precedence behavior
        assert_ne!(
            parse_single_unwrap(arena, "(0 + 1) ; T"),
            parse_single_unwrap(arena, "0 + 1 ; T")
        ); // Default: 0 + (1 ; T)

        assert_ne!(
            parse_single_unwrap(arena, "0 + (1 ; T)"),
            parse_single_unwrap(arena, "0 ; 1 + T")
        ); // Default: (0 ; 1) + T

        assert_ne!(
            parse_single_unwrap(arena, "(~0)*"),
            parse_single_unwrap(arena, "~0*")
        ); // Default: ~(0*)

        assert_ne!(
            parse_single_unwrap(arena, "(0 U 1) + T"),
            parse_single_unwrap(arena, "0 U 1 + T")
        ); // Default: 0 U (1 + T)
    }

    #[test]
    fn test_field_operations_precedence() {
        let arena = &bumpalo::Bump::new();
        // Field operations should be atomic (highest precedence)
        assert_eq!(
            parse_single_unwrap(arena, "x1 := 0 + x2 == 1"),
            parse_single_unwrap(arena, "(x1 := 0) + (x2 == 1)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "~x1 := 0"),
            parse_single_unwrap(arena, "~(x1 := 0)")
        );
        assert_eq!(
            parse_single_unwrap(arena, "x1 == 1*"),
            parse_single_unwrap(arena, "(x1 == 1)*")
        );
    }

    #[test]
    fn test_precedence_edge_cases() {
        let arena = &bumpalo::Bump::new();
        // Edge cases that might be tricky

        // Prefix followed by postfix should work
        assert_eq!(
            parse_single_unwrap(arena, "~T*"),
            parse_single_unwrap(arena, "~(T*)")
        );

        // Multiple prefix operators
        assert_eq!(
            parse_single_unwrap(arena, "~~X F 0"),
            parse_single_unwrap(arena, "~(~(X (F 0)))")
        );

        // Mixing all operator types
        assert_eq!(
            parse_single_unwrap(arena, "~(x1:=0)* + T ; dup & 1 U 0"),
            parse_single_unwrap(arena, "(~(x1:=0)* + (T ; (dup & 1))) U 0")
        );
    }

    #[test]
    fn test_original_user_issue() {
        let arena = &bumpalo::Bump::new();
        // Test the specific issue reported by the user:
        // (a ; b + c) should be parsed as (a ; b) + c, not a ; (b + c)

        // Using concrete expressions to make the test clearer
        assert_eq!(
            parse_single_unwrap(arena, "x1==0 ; x2==1 + T"),
            parse_single_unwrap(arena, "(x1==0 ; x2==1) + T")
        );

        // Verify the opposite case with parentheses
        assert_eq!(
            parse_single_unwrap(arena, "x1==0 ; (x2==1 + T)"),
            parse_single_unwrap(arena, "x1==0 ; (x2==1 + T)")
        ); // Already correct

        // They should be different expressions
        assert_ne!(
            parse_single_unwrap(arena, "x1==0 ; x2==1 + T"),
            parse_single_unwrap(arena, "x1==0 ; (x2==1 + T)")
        );

        // More examples showing the fix
        assert_eq!(
            parse_single_unwrap(arena, "0 ; 1 + T"),
            parse_single_unwrap(arena, "(0 ; 1) + T")
        );
        assert_eq!(
            parse_single_unwrap(arena, "dup ; T ^ 1"),
            parse_single_unwrap(arena, "(dup ; T) ^ 1")
        );
        assert_eq!(
            parse_single_unwrap(arena, "T ; 0 - 1"),
            parse_single_unwrap(arena, "(T ; 0) - 1")
        );
    }

    #[test]
    fn test_bit_range_parsing() {
        let arena = &bumpalo::Bump::new();
        // Test parsing bit range assignments
        assert_eq!(
            parse_single_unwrap(arena, "x[0..8] := 255"),
            Expr::bit_range_assign(
                arena,
                0,
                8,
                vec![true, true, true, true, true, true, true, true]
            )
        );

        // Test parsing bit range pattern matches
        assert_eq!(
            parse_single_unwrap(arena, "x[2..4] ~ 3"),
            Expr::bit_range_match(arena, 2, 4, Pattern::Exact(&[true, true]))
        );

        // Test with zero
        assert_eq!(
            parse_single_unwrap(arena, "x[0..4] ~ 0"),
            Expr::bit_range_match(arena, 0, 4, Pattern::Exact(&[false, false, false, false]))
        );

        // Test single bit range
        assert_eq!(
            parse_single_unwrap(arena, "x[5..6] := 1"),
            Expr::bit_range_assign(arena, 5, 6, vec![true])
        );
    }

    #[test]
    fn test_bit_range_in_expression() {
        let arena = &bumpalo::Bump::new();
        // Test bit ranges in larger expressions
        let expr = parse_single_unwrap(arena, "x[0..4] ~ 5 + x[4..8] := 10");
        let expected = Expr::union(
            arena,
            Expr::bit_range_match(arena, 0, 4, Pattern::Exact(&[true, false, true, false])),
            Expr::bit_range_assign(arena, 4, 8, vec![false, true, false, true]),
        );
        assert_eq!(expr, expected);

        // Test with sequence
        let expr2 = parse_single_unwrap(arena, "x[0..2] := 3 ; x[2..4] ~ 0");
        let expected2 = Expr::sequence(
            arena,
            Expr::bit_range_assign(arena, 0, 2, vec![true, true]),
            Expr::bit_range_match(arena, 2, 4, Pattern::Exact(&[false, false])),
        );
        assert_eq!(expr2, expected2);
    }

    #[test]
    fn test_literal_formats() {
        let arena = &bumpalo::Bump::new();
        // Test binary literals
        let expr1 = parse_single_unwrap(arena, "x[0..4] := 0b1010");
        assert_eq!(
            expr1,
            Expr::bit_range_assign(arena, 0, 4, vec![false, true, false, true])
        );

        // Test hexadecimal literals
        let expr2 = parse_single_unwrap(arena, "x[0..8] := 0xFF");
        assert_eq!(
            expr2,
            Expr::bit_range_assign(
                arena,
                0,
                8,
                vec![true, true, true, true, true, true, true, true]
            )
        );

        // Test IP address literals (192.168.1.1 = 0xC0A80101)
        let expr3 = parse_single_unwrap(arena, "x[0..32] := 192.168.1.1");
        // IP addresses are converted in big-endian format: 192.168.1.1 = 0xC0A80101
        // But our bit vector is little-endian, so bit 0 is the LSB
        // 0xC0A80101 = 3232235777 in decimal
        let ip_num = 0xC0A80101u32;
        let ip_bits: Vec<bool> = (0..32).map(|i| (ip_num >> i) & 1 == 1).collect();
        assert_eq!(expr3, Expr::bit_range_assign(arena, 0, 32, ip_bits));

        // Test mixed formats in compound expression
        let expr4 = parse_single_unwrap(arena, "x[0..4] ~ 0b1100 + x[4..12] := 0xF0");
        let expected4 = Expr::union(
            arena,
            Expr::bit_range_match(arena, 0, 4, Pattern::Exact(&[false, false, true, true])),
            Expr::bit_range_assign(
                arena,
                4,
                12,
                vec![false, false, false, false, true, true, true, true],
            ),
        );
        assert_eq!(expr4, expected4);
    }

    #[test]
    fn test_literal_edge_cases() {
        let arena = &bumpalo::Bump::new();
        // Test single bits
        assert_eq!(
            parse_single_unwrap(arena, "x[0..1] := 0b1"),
            Expr::bit_range_assign(arena, 0, 1, vec![true])
        );

        assert_eq!(
            parse_single_unwrap(arena, "x[5..6] ~ 0"),
            Expr::bit_range_match(arena, 5, 6, Pattern::Exact(&[false]))
        );

        // Test larger hex values
        assert_eq!(
            parse_single_unwrap(arena, "x[0..16] := 0xABCD"),
            Expr::bit_range_assign(
                arena,
                0,
                16,
                vec![
                    true, false, true, true, false, false, true, true, // 0xCD = 205
                    true, true, false, true, false, true, false, true // 0xAB = 171
                ]
            )
        );
    }

    #[test]
    fn test_bit_range_number_ranges() {
        let arena = &bumpalo::Bump::new();
        // Test range patterns starting with 0 and 1 like x[0..3] ~ 1-3
        let expr1 = parse_single_unwrap(arena, "x[0..3] ~ 1-3");
        let expected1 = Expr::bit_range_match(
            arena,
            0,
            3,
            Pattern::IpRange {
                start: &[true, false, false], // 1 in 3 bits (LSB first)
                end: &[true, true, false],    // 3 in 3 bits (LSB first)
            },
        );
        assert_eq!(expr1, expected1);

        // Test range starting with 0
        let expr2 = parse_single_unwrap(arena, "x[0..4] ~ 0-7");
        let expected2 = Expr::bit_range_match(
            arena,
            0,
            4,
            Pattern::IpRange {
                start: &[false, false, false, false], // 0 in 4 bits
                end: &[true, true, true, false],      // 7 in 4 bits (LSB first)
            },
        );
        assert_eq!(expr2, expected2);

        // Test range with single bit
        let expr3 = parse_single_unwrap(arena, "x[5..6] ~ 0-1");
        let expected3 = Expr::bit_range_match(
            arena,
            5,
            6,
            Pattern::IpRange {
                start: &[false], // 0 in 1 bit
                end: &[true],    // 1 in 1 bit
            },
        );
        assert_eq!(expr3, expected3);

        // Test the specific requested case: x[0..3] ~ 1-4 (needs 4 bits for range 1-4)
        let expr4 = parse_single_unwrap(arena, "x[0..4] ~ 1-4");
        let expected4 = Expr::bit_range_match(
            arena,
            0,
            4,
            Pattern::IpRange {
                start: &[true, false, false, false], // 1 in 4 bits (LSB first)
                end: &[false, false, true, false],   // 4 in 4 bits (LSB first)
            },
        );
        assert_eq!(expr4, expected4);
    }

    #[test]
    fn test_let_parsing() {
        let arena = &bumpalo::Bump::new();
        // Basic let binding
        let expr = parse_single_unwrap(arena, "let x = 1 in x + 0");
        assert_eq!(
            expr,
            Expr::let_in(
                arena,
                "x",
                Expr::one(arena),
                Expr::union(arena, Expr::var(arena, "x"), Expr::zero(arena))
            )
        );

        // Let with assignment
        let expr = parse_single_unwrap(arena, "let config = x0 := 1 in config ; x1 := 0");
        assert_eq!(
            expr,
            Expr::let_in(
                arena,
                "config",
                Expr::assign(arena, 0, true),
                Expr::sequence(
                    arena,
                    Expr::var(arena, "config"),
                    Expr::assign(arena, 1, false)
                )
            )
        );

        // Let with test
        let expr = parse_single_unwrap(arena, "let test = x0 == 1 in test & x1 == 0");
        assert_eq!(
            expr,
            Expr::let_in(
                arena,
                "test",
                Expr::test(arena, 0, true),
                Expr::intersect(arena, Expr::var(arena, "test"), Expr::test(arena, 1, false))
            )
        );
    }

    #[test]
    fn test_nested_let_parsing() {
        let arena = &bumpalo::Bump::new();
        // Nested let bindings
        let expr = parse_single_unwrap(arena, "let x = 0 in let y = 1 in x + y");
        assert_eq!(
            expr,
            Expr::let_in(
                arena,
                "x",
                Expr::zero(arena),
                Expr::let_in(
                    arena,
                    "y",
                    Expr::one(arena),
                    Expr::union(arena, Expr::var(arena, "x"), Expr::var(arena, "y"))
                )
            )
        );

        // Let with complex expression
        let expr = parse_single_unwrap(arena, "let p = (x0 := 1 ; x1 := 0) in p + p*");
        assert_eq!(
            expr,
            Expr::let_in(
                arena,
                "p",
                Expr::sequence(
                    arena,
                    Expr::assign(arena, 0, true),
                    Expr::assign(arena, 1, false)
                ),
                Expr::union(
                    arena,
                    Expr::var(arena, "p"),
                    Expr::star(arena, Expr::var(arena, "p"))
                )
            )
        );
    }

    #[test]
    fn test_let_with_various_names() {
        let arena = &bumpalo::Bump::new();
        // Let with various valid variable names
        let expr = parse_single_unwrap(arena, "let myVar = 1 in myVar");
        assert_eq!(
            expr,
            Expr::let_in(arena, "myVar", Expr::one(arena), Expr::var(arena, "myVar"))
        );

        let expr = parse_single_unwrap(arena, "let test_var = 0 in test_var");
        assert_eq!(
            expr,
            Expr::let_in(
                arena,
                "test_var",
                Expr::zero(arena),
                Expr::var(arena, "test_var")
            )
        );

        let expr = parse_single_unwrap(arena, "let v123 = x0 := 1 in v123");
        assert_eq!(
            expr,
            Expr::let_in(
                arena,
                "v123",
                Expr::assign(arena, 0, true),
                Expr::var(arena, "v123")
            )
        );
    }

    #[test]
    fn test_let_precedence() {
        let arena = &bumpalo::Bump::new();
        // Let has low precedence, binds to the right
        let expr = parse_single_unwrap(arena, "0 + let x = 1 in x");
        assert_eq!(
            expr,
            Expr::union(
                arena,
                Expr::zero(arena),
                Expr::let_in(arena, "x", Expr::one(arena), Expr::var(arena, "x"))
            )
        );

        // Parentheses override precedence
        let expr = parse_single_unwrap(arena, "(let x = 1 in x) + 0");
        assert_eq!(
            expr,
            Expr::union(
                arena,
                Expr::let_in(arena, "x", Expr::one(arena), Expr::var(arena, "x")),
                Expr::zero(arena)
            )
        );
    }

    #[test]
    fn test_let_errors() {
        let arena = &bumpalo::Bump::new();
        // Missing 'in' keyword
        assert!(parse_all(arena, "let x = 1 x").is_err());

        // Missing body
        assert!(parse_all(arena, "let x = 1 in").is_err());

        // Missing definition
        assert!(parse_all(arena, "let x = in 1").is_err());

        // Missing variable name
        assert!(parse_all(arena, "let = 1 in 0").is_err());

        // Invalid variable name
        assert!(parse_all(arena, "let 123 = 1 in 0").is_err());
    }

    #[test]
    fn test_bit_range_alias_parsing() {
        let arena = &bumpalo::Bump::new();
        // Basic bit range alias
        let expr = parse_single_unwrap(arena, "let ip = &x[0..32] in ip := 192.168.1.1");
        // IP address 192.168.1.1 in little-endian bit representation
        // Using the same conversion as in test_literal_formats
        let ip_num = 0xC0A80101u32; // 192.168.1.1 in hex
        let ip_bits: Vec<bool> = (0..32).map(|i| (ip_num >> i) & 1 == 1).collect();
        assert_eq!(
            expr,
            Expr::let_bit_range(arena, "ip", 0, 32, Expr::var_assign(arena, "ip", ip_bits))
        );

        // Bit range alias with test
        let expr = parse_single_unwrap(arena, "let port = &x[32..48] in port ~ 80");
        let port_bits = vec![false, false, false, false, true, false, true]; // 80 in 7 bits (minimal)
        assert_eq!(
            expr,
            Expr::let_bit_range(
                arena,
                "port",
                32,
                48,
                Expr::var_match(
                    arena,
                    "port",
                    Pattern::Exact(arena.alloc_slice_copy(&port_bits))
                )
            )
        );

        // Multiple bit range aliases
        let expr = parse_single_unwrap(
            arena,
            "let src = &x[0..32] in let dst = &x[32..64] in src ~ 10.0.0.1 & dst ~ 10.0.0.2",
        );
        let ip1_num = 0x0A000001u32; // 10.0.0.1 in hex
        let ip2_num = 0x0A000002u32; // 10.0.0.2 in hex
        // Generate LSB-first order to match ip_to_bits
        let ip1_bits: Vec<bool> = (0..32).map(|i| (ip1_num >> i) & 1 == 1).collect();
        let ip2_bits: Vec<bool> = (0..32).map(|i| (ip2_num >> i) & 1 == 1).collect();
        assert_eq!(
            expr,
            Expr::let_bit_range(
                arena,
                "src",
                0,
                32,
                Expr::let_bit_range(
                    arena,
                    "dst",
                    32,
                    64,
                    Expr::intersect(
                        arena,
                        Expr::var_match(
                            arena,
                            "src",
                            Pattern::Exact(arena.alloc_slice_copy(&ip1_bits))
                        ),
                        Expr::var_match(
                            arena,
                            "dst",
                            Pattern::Exact(arena.alloc_slice_copy(&ip2_bits))
                        )
                    )
                )
            )
        );
    }

    #[test]
    fn test_alias_with_regular_let() {
        let arena = &bumpalo::Bump::new();
        // Mix of regular let and bit range alias
        let expr = parse_single_unwrap(
            arena,
            "let config = x0 := 1 in let ip = &x[0..32] in config ; ip := 192.168.1.1",
        );
        let ip_num = 0xC0A80101u32; // 192.168.1.1 in hex
        let ip_bits: Vec<bool> = (0..32).map(|i| (ip_num >> i) & 1 == 1).collect();
        assert_eq!(
            expr,
            Expr::let_in(
                arena,
                "config",
                Expr::assign(arena, 0, true),
                Expr::let_bit_range(
                    arena,
                    "ip",
                    0,
                    32,
                    Expr::sequence(
                        arena,
                        Expr::var(arena, "config"),
                        Expr::var_assign(arena, "ip", ip_bits)
                    )
                )
            )
        );
    }

    #[test]
    fn test_alias_errors() {
        let arena = &bumpalo::Bump::new();
        // Missing ampersand
        assert!(parse_all(arena, "let ip = x[0..32] in ip := 1").is_err());

        // Missing brackets
        assert!(parse_all(arena, "let ip = &x in ip := 1").is_err());

        // Missing range
        assert!(parse_all(arena, "let ip = &x[] in ip := 1").is_err());

        // Invalid range syntax
        assert!(parse_all(arena, "let ip = &x[0-32] in ip := 1").is_err());

        // Non-x field reference
        assert!(parse_all(arena, "let ip = &y[0..32] in ip := 1").is_err());

        // Missing 'in' keyword
        assert!(parse_all(arena, "let ip = &x[0..32] ip := 1").is_err());
    }

    #[test]
    fn test_alias_precedence() {
        let arena = &bumpalo::Bump::new();
        // Alias in larger expression
        let expr = parse_single_unwrap(arena, "0 + let ip = &x[0..32] in ip ~ 10.0.0.1");
        let ip_num = 0x0A000001u32; // 10.0.0.1 in hex
        // Generate LSB-first order to match ip_to_bits
        let ip_bits: Vec<bool> = (0..32).map(|i| (ip_num >> i) & 1 == 1).collect();
        assert_eq!(
            expr,
            Expr::union(
                arena,
                Expr::zero(arena),
                Expr::let_bit_range(
                    arena,
                    "ip",
                    0,
                    32,
                    Expr::var_match(
                        arena,
                        "ip",
                        Pattern::Exact(arena.alloc_slice_copy(&ip_bits))
                    )
                )
            )
        );
    }

    // ===== ERROR MESSAGE TESTS =====
    #[test]
    fn test_helpful_error_messages_for_disallowed_equals() {
        let arena = &bumpalo::Bump::new();
        // Test that we get helpful error messages when using == incorrectly

        // Variable with == should suggest ~
        let result = parse_all(arena, "let port = &x[0..16] in port == 1024");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("The '==' operator is only supported for simple field tests"));
        assert!(error.contains("use the '~' operator instead"));
        assert!(error.contains("port ~ 1024"));

        // Bit range with == should suggest ~
        let result = parse_all(arena, "x[0..8] == 255");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Expected ':=' or '~' after bit range"));

        // IP address with variable == should suggest ~
        let result = parse_all(arena, "let ip = &x[0..32] in ip == 192.168.1.1");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("The '==' operator is only supported for simple field tests"));

        // Multiple variables with == should suggest ~
        let result = parse_all(
            arena,
            "let a = &x[0..8] in let b = &x[8..16] in a == 10 & b == 20",
        );
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("The '==' operator is only supported for simple field tests"));
    }

    #[test]
    fn test_allowed_equals_still_work() {
        let arena = &bumpalo::Bump::new();
        // Test that simple field tests with == still work

        // Single bit tests should still work
        assert!(parse_all(arena, "x0 == 0").is_ok());
        assert!(parse_all(arena, "x1 == 1").is_ok());
        assert!(parse_all(arena, "x42 == 0").is_ok());

        // Combinations of simple field tests should work
        assert!(parse_all(arena, "x0 == 1 & x1 == 0").is_ok());
        assert!(parse_all(arena, "(x0 == 1; x1 := 0) + (x0 == 0; x1 := 1)").is_ok());

        // Mixed with other operations should work
        assert!(parse_all(arena, "x0 == 1; x[8..16] ~ 255").is_ok());
        assert!(parse_all(arena, "let port = &x[0..16] in x0 == 1 & port ~ 80").is_ok());
    }

    #[test]
    fn test_error_message_context() {
        let arena = &bumpalo::Bump::new();
        // Test error messages in different contexts

        // In sequence
        let result = parse_all(arena, "x0 := 1; let port = &x[0..16] in port == 80");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("The '==' operator is only supported for simple field tests"));

        // In union
        let result = parse_all(arena, "x0 == 1 + let port = &x[0..16] in port == 80");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("The '==' operator is only supported for simple field tests"));

        // In intersection
        let result = parse_all(arena, "x0 == 1 & let port = &x[0..16] in port == 80");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("The '==' operator is only supported for simple field tests"));

        // Nested in let
        let result = parse_all(
            arena,
            "let outer = x0 == 1 in let port = &x[0..16] in port == 80",
        );
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("The '==' operator is only supported for simple field tests"));
    }

    #[test]
    fn test_specific_suggestions_in_error_messages() {
        let arena = &bumpalo::Bump::new();
        // Test that error messages contain specific helpful suggestions

        // Hexadecimal values
        let result = parse_all(arena, "let byte = &x[0..8] in byte == 0xFF");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("use the '~' operator instead"));

        // Binary values
        let result = parse_all(arena, "x[0..4] == 0b1010");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Expected ':=' or '~' after bit range"));

        // IP addresses
        let result = parse_all(arena, "let src = &x[0..32] in src == 10.0.0.1");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("The '==' operator is only supported for simple field tests"));
        assert!(error.contains("x0 == 1"));

        // Large numbers
        let result = parse_all(arena, "let port = &x[0..16] in port == 65535");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("port ~ 1024"));
    }
}

// Temporary function to satisfy lib.rs if it's calling something specific not yet refactored.
// pub fn old_tokenize_placeholder(input: &str) -> Result<Vec<super::Token>, String> {
//     Err("Tokenizer placeholder")
// }
