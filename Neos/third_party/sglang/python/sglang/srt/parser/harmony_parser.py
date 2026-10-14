import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple


@dataclass
class Event:
    """Represents a parsed event from the Harmony stream."""

    event_type: str
    content: str
    raw_text: str = None  


@dataclass
class Token:
    """A structural token in the Harmony format."""

    type: str
    start: int
    end: int


def prefix_hold(text: str, tokens: List[str]) -> Tuple[str, str]:
    """
    Holds back the longest suffix of `text` that could be a prefix of any token.
    Returns (emit_now, keep_for_later).
    """
    if not text:
        return "", ""
    max_hold = 0
    for tok in tokens:
        if not tok:
            continue
        
        L = min(len(tok) - 1, len(text))
        for k in range(L, 0, -1):
            if tok.startswith(text[-k:]):
                max_hold = max(max_hold, k)
                break
    if max_hold == 0:
        return text, ""
    return text[:-max_hold], text[-max_hold:]


def iter_tokens(text: str, start_pos: int = 0) -> Iterator[Token]:
    """Iterate over structural tokens in left-to-right order."""
    TOKENS = {
        "<|start|>": "START",
        "<|channel|>": "CHANNEL",
        "<|message|>": "MESSAGE",
        "<|constrain|>": "CONSTRAIN",
        "<|end|>": "END",
        "<|call|>": "CALL",
        "<|return|>": "RETURN",
    }

    pos = start_pos
    has_unknown_tokens = False
    while pos < len(text):
        
        marker_pos = text.find("<|", pos)
        if marker_pos == -1:
            break

        
        if marker_pos > pos:
            yield Token("TEXT", pos, marker_pos)

        
        found_token = False

        for literal, token_type in TOKENS.items():
            if text.startswith(literal, marker_pos):
                yield Token(token_type, marker_pos, marker_pos + len(literal))
                pos = marker_pos + len(literal)
                found_token = True
                break
        if not found_token:
            tail = text[marker_pos:]
            is_partial = any(lit.startswith(tail) for lit in TOKENS)
            if is_partial:
                
                yield Token("TEXT", marker_pos, len(text))
                pos = len(text)
                break
            else:
                
                has_unknown_tokens = True
                
                yield Token("TEXT", marker_pos, marker_pos + 2)

                
                close_pos = text.find("|>", marker_pos + 2)
                if close_pos != -1:
                    
                    next_marker = text.find("<|", close_pos + 2)
                    if next_marker != -1:
                        
                        yield Token("TEXT", marker_pos + 2, next_marker)
                        pos = next_marker
                    else:
                        
                        yield Token("TEXT", marker_pos + 2, len(text))
                        pos = len(text)
                        break
                else:
                    
                    pos = marker_pos + 2

    
    if pos < len(text):
        yield Token("TEXT", pos, len(text))
    elif pos == len(text) and has_unknown_tokens:
        
        
        for literal in TOKENS.keys():
            if text.endswith(literal):
                yield Token("TEXT", pos, pos)
                break


class CanonicalStrategy:
    """Parses the canonical Harmony format with channel markers."""

    def __init__(self):
        self.guard_tokens = [
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|end|>",
            "<|call|>",
            "<|return|>",
        ]

    def parse(self, text: str) -> Tuple[List[Event], str]:
        events = []
        tokens = list(iter_tokens(text))

        if not tokens:
            return events, ""

        pos = 0
        while pos < len(tokens):
            token = tokens[pos]

            if token.type == "TEXT":
                
                if pos == len(tokens) - 1:  
                    emit, hold = prefix_hold(
                        text[token.start : token.end], self.guard_tokens
                    )
                    if emit:
                        events.append(Event("normal", emit))
                    return events, hold
                else:
                    
                    if self._is_commentary_filler_between_blocks(text, tokens, pos):
                        
                        pos += 1
                    else:
                        content = text[token.start : token.end]
                        
                        if not self._is_standalone_structural_token(content):
                            events.append(Event("normal", content))
                        pos += 1

            elif token.type in ("START", "CHANNEL"):
                
                block_result = self._parse_block(text, tokens, pos)
                if block_result is None:
                    
                    partial_result = self._parse_partial_analysis(text, tokens, pos)
                    if partial_result:
                        event, remaining_text = partial_result
                        events.append(event)
                        return events, remaining_text
                    
                    remaining_start = tokens[pos].start
                    return events, text[remaining_start:]
                event, new_pos = block_result
                if event:
                    events.append(event)
                pos = new_pos

            else:
                
                if self._is_commentary_filler_between_blocks(text, tokens, pos):
                    
                    pos += 1
                else:
                    
                    content = text[token.start : token.end]
                    if not self._is_standalone_structural_token(content):
                        events.append(Event("normal", content))
                    pos += 1

        return events, ""

    def _parse_partial_analysis(
        self, text: str, tokens: List[Token], start_pos: int
    ) -> Optional[Tuple[Event, str]]:
        """Try to parse partial analysis content for incremental streaming."""
        pos = start_pos

        
        if pos < len(tokens) and tokens[pos].type == "START":
            pos += 1

        
        channel_pos = None
        message_pos = None

        for i in range(pos, len(tokens)):
            if tokens[i].type == "CHANNEL" and channel_pos is None:
                channel_pos = i
            elif tokens[i].type == "MESSAGE":
                message_pos = i
                break

        if channel_pos is None or message_pos is None:
            return None

        
        channel_start = (
            tokens[channel_pos + 1].start
            if channel_pos + 1 < len(tokens)
            else tokens[channel_pos].end
        )
        channel_end = tokens[message_pos].start
        channel_header = text[channel_start:channel_end]

        channel_type = self._extract_channel_type(channel_header)
        if channel_type != "analysis":
            return None  

        
        content_start = tokens[message_pos].end
        content = text[content_start:]

        
        remaining_text = text[tokens[start_pos].start : content_start]
        return Event("reasoning", content), remaining_text

    def _extract_channel_type(self, header_text: str) -> Optional[str]:
        """Extract channel type from header, ignoring other attributes like to=... or <|constrain|>..."""
        
        header_clean = header_text.strip()

        if header_clean.lower().startswith("analysis"):
            return "analysis"
        elif header_clean.lower().startswith("commentary"):
            return "commentary"
        elif header_clean.lower().startswith("final"):
            return "final"
        else:
            return None  

    def _parse_block(
        self, text: str, tokens: List[Token], start_pos: int
    ) -> Optional[Tuple[Optional[Event], int]]:
        """Parse a channel block. Returns (event, next_pos) or None if incomplete."""
        pos = start_pos

        
        if pos < len(tokens) and tokens[pos].type == "START":
            pos += 1

        
        channel_pos = None
        message_pos = None

        for i in range(pos, len(tokens)):
            if tokens[i].type == "CHANNEL" and channel_pos is None:
                channel_pos = i
            elif tokens[i].type == "MESSAGE":
                message_pos = i
                break

        if message_pos is None:
            return None  

        
        if channel_pos is None:
            content_start = tokens[message_pos].end
            
            end_token_pos = None
            for i in range(message_pos + 1, len(tokens)):
                if tokens[i].type in ("END", "CALL", "RETURN"):
                    end_token_pos = i
                    break
            if end_token_pos is None:
                return None  
            content = text[content_start : tokens[end_token_pos].start]
            return Event("normal", content), end_token_pos + 1

        
        pos = channel_pos + 1  

        
        channel_start = tokens[pos].start if pos < len(tokens) else tokens[pos - 1].end
        channel_end = tokens[message_pos].start
        channel_header = text[channel_start:channel_end]

        channel_type = self._extract_channel_type(channel_header)
        if not channel_type:
            return None  

        pos = message_pos + 1  

        
        content_start = tokens[message_pos].end
        end_pos = pos

        
        if channel_type == "final":
            while end_pos < len(tokens) and tokens[end_pos].type != "RETURN":
                end_pos += 1
        elif channel_type == "analysis":
            while end_pos < len(tokens) and tokens[end_pos].type not in ("END", "CALL"):
                end_pos += 1
        else:  
            while end_pos < len(tokens) and tokens[end_pos].type not in ("END", "CALL"):
                end_pos += 1

        if end_pos >= len(tokens):
            
            if channel_type == "final":
                
                content = text[content_start:]
                return Event("normal", content), end_pos
            return None  

        end_token = tokens[end_pos]
        content = text[content_start : end_token.start]

        
        if channel_type == "analysis":
            if end_token.type == "CALL":
                
                raw_text = text[tokens[start_pos].start : end_token.end]
                return Event("tool_call", content.strip(), raw_text), end_pos + 1
            else:
                return Event("reasoning", content), end_pos + 1
        elif channel_type == "commentary":
            if end_token.type == "CALL":
                raw_text = text[tokens[start_pos].start : end_token.end]
                return Event("tool_call", content.strip(), raw_text), end_pos + 1
            else:
                return Event("normal", content), end_pos + 1
        elif channel_type == "final":
            
            final_content = content
            if end_token.type == "RETURN" and end_pos + 1 < len(tokens):
                next_token = tokens[end_pos + 1]
                if next_token.type == "TEXT":
                    final_content += text[next_token.start : next_token.end]
                    return Event("normal", final_content), end_pos + 2
            return Event("normal", final_content), end_pos + 1

        return None, end_pos + 1

    def _is_commentary_filler_between_blocks(
        self, text: str, tokens: List[Token], pos: int
    ) -> bool:
        """Check if this is commentary filler text or problematic structural tokens in malformed sequences."""
        current_token = tokens[pos]
        current_text = text[current_token.start : current_token.end].strip()

        
        if pos > 0 and pos + 1 < len(tokens):
            prev_token = tokens[pos - 1]
            next_token = tokens[pos + 1]

            
            if (
                prev_token.type == "CALL"
                and next_token.type == "CHANNEL"
                and current_text.lower() == "commentary"
            ):
                return True

        
        if pos > 0:
            prev_token = tokens[pos - 1]

            
            
            if prev_token.type == "CALL":
                
                if current_token.type == "MESSAGE":
                    return True

                
                if (
                    current_token.type == "TEXT"
                    and current_text.lower() == "commentary"
                ):
                    return True

        return False

    def _is_standalone_structural_token(self, content: str) -> bool:
        """Check if content is just a standalone structural token that should be filtered."""
        content_stripped = content.strip()
        structural_tokens = [
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|end|>",
            "<|call|>",
            "<|return|>",
        ]
        return content_stripped in structural_tokens


class TextStrategy:
    """Parses the text-based Harmony fallback format."""

    def __init__(self):
        self.buffer_context = ""
        self.patterns = {
            "analysis_then_final": re.compile(
                r"^\s*(?:assistant)?\s*(analysis|commentary)(.*?)\s*assistantfinal\s*(.*)\s*$",
                re.IGNORECASE | re.DOTALL,
            ),
            "final_only": re.compile(
                r"^\s*assistantfinal\s*(.*)\s*$", re.IGNORECASE | re.DOTALL
            ),
            "analysis_only": re.compile(
                r"^\s*(?:assistant)?\s*(analysis|commentary)(.*)\s*$",
                re.IGNORECASE | re.DOTALL,
            ),
        }

    def set_buffer_context(self, buffer: str):
        self.buffer_context = buffer

    def parse(self, text: str) -> Tuple[List[Event], str]:
        events = []

        m = self.patterns["analysis_then_final"].match(text)
        if m:
            channel, reasoning, final = m.groups()
            if channel.lower() == "analysis" and reasoning.strip():
                events.append(Event("reasoning", reasoning.strip()))
            elif channel.lower() == "commentary" and reasoning.strip():
                events.append(Event("normal", reasoning.strip()))
            if final.strip():
                events.append(Event("normal", final.strip()))
            return events, ""

        
        if re.search(
            r"(?:^|\s)(?:assistant)?\s*(analysis|commentary)", text, re.IGNORECASE
        ):
            low = text.lower()
            if "assistantfin" in low and "assistantfinal" not in low:
                return events, text

        m = self.patterns["final_only"].match(text)
        if m:
            final = m.group(1)
            if final.strip():
                events.append(Event("normal", final.strip()))
            return events, ""

        m = self.patterns["analysis_only"].match(text)
        if m:
            channel, content = m.groups()
            emit, hold = prefix_hold(content, ["assistantfinal"])
            if channel.lower() == "analysis" and emit:
                
                events.append(Event("reasoning", emit))
                
                
                
                if hold:
                    return events, text[: m.start(2)] + hold
                else:
                    return events, channel
            elif channel.lower() == "commentary" and emit:
                
                content_out = emit if hold else emit.strip()
                events.append(Event("normal", content_out))
                if hold:
                    return events, text[: m.start(2)] + hold
                else:
                    return events, ""
            
            return events, text[: m.start(2)] + hold

        emit, hold = prefix_hold(text, ["analysis", "commentary", "assistantfinal"])
        if emit:
            events.append(Event("normal", emit))
        return events, hold


class HarmonyParser:
    """Facade for parsing Harmony format, switching between strategies."""

    def __init__(self):
        self.strategy = None
        self._buffer = ""
        self._should_filter_commentary = (
            False  
        )
        self._partial_commentary = (
            ""  
        )

    def parse(self, chunk: str) -> List[Event]:
        self._buffer += chunk

        if self.strategy is None:
            if "<|channel|>" in self._buffer or "<|start|>" in self._buffer:
                self.strategy = CanonicalStrategy()
            elif re.search(
                r"(?:^|\s)(?:assistant)?\s*(analysis|commentary|assistantfinal)",
                self._buffer,
                re.IGNORECASE,
            ):
                self.strategy = TextStrategy()
            else:
                
                return []

        if hasattr(self.strategy, "set_buffer_context"):
            
            self.strategy.set_buffer_context(self._buffer)

        events, remaining = self.strategy.parse(self._buffer)

        
        buffer_has_call_token = self._buffer.rstrip().endswith("<|call|>")

        self._buffer = remaining

        
        filtered_events = []
        for event in events:
            should_filter = False

            if event.event_type == "normal":
                
                if self._should_filter_commentary or self._partial_commentary:
                    
                    potential_commentary = (
                        self._partial_commentary + event.content.strip().lower()
                    )

                    if potential_commentary == "commentary":
                        
                        should_filter = True
                        self._partial_commentary = ""  
                        self._should_filter_commentary = False  
                    elif "commentary".startswith(potential_commentary):
                        
                        should_filter = True
                        self._partial_commentary = potential_commentary
                    else:
                        
                        self._partial_commentary = ""
                        self._should_filter_commentary = False
                else:
                    
                    self._partial_commentary = ""

            if should_filter:
                
                continue

            
            if event.event_type == "tool_call":
                self._should_filter_commentary = (
                    True  
                )
                self._partial_commentary = ""  
            elif buffer_has_call_token:
                self._should_filter_commentary = (
                    True  
                )

            filtered_events.append(event)

        return filtered_events
