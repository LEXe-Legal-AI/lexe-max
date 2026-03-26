"""Generic Document Chunker for Library (Private Documents).

Sliding-window chunker with paragraph/sentence awareness and configurable overlap.
Designed for user-uploaded documents (contracts, legal opinions, memos) — NOT
legislative text (use MarkerChunker for that).

Usage:
    chunker = GenericChunker(target_size=1000, overlap_chars=150)
    chunks = chunker.chunk(plain_text)

    for c in chunks:
        print(f"Chunk {c.chunk_no}: {c.char_start}-{c.char_end} ({c.token_est} tokens)")
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# Sentence boundary pattern: period/question/exclamation followed by space,
# or semicolon followed by newline.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!])\s+|(?<=;)\n")


@dataclass
class ChunkResult:
    """A single chunk produced by GenericChunker."""

    chunk_no: int
    char_start: int       # Inclusive offset in original text
    char_end: int         # Exclusive offset in original text
    text: str
    token_est: int

    def __post_init__(self) -> None:
        if self.token_est <= 0:
            self.token_est = max(1, len(self.text) // 4)


class GenericChunker:
    """Sliding-window chunker with paragraph and sentence awareness.

    Algorithm:
        1. Split text into paragraphs on ``\\n\\n``.
        2. Accumulate paragraphs into a buffer until *target_size* is reached.
        3. Finalise the chunk and start the next one with *overlap_chars*
           characters carried over from the end of the previous chunk.
        4. If a single paragraph exceeds *max_size*, split it on sentence
           boundaries.
        5. If a single sentence still exceeds *max_size*, force-split at
           *max_size* while preserving *overlap_chars*.

    Parameters:
        target_size:   Ideal chunk length in characters (800-1200 range).
        overlap_chars: Number of characters that consecutive chunks share
                       at their boundary. **Must be > 0.**
        max_size:      Hard ceiling — any segment longer is force-split.
        min_size:      Chunks shorter than this are merged with neighbours.
    """

    def __init__(
        self,
        target_size: int = 1000,
        overlap_chars: int = 150,
        max_size: int = 2000,
        min_size: int = 20,
    ) -> None:
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")
        if overlap_chars >= target_size:
            raise ValueError("overlap_chars must be < target_size")

        self.target_size = target_size
        self.overlap_chars = overlap_chars
        self.max_size = max_size
        self.min_size = min_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[ChunkResult]:
        """Split *text* into overlapping chunks.

        Returns an empty list when *text* is empty or shorter than *min_size*.
        """
        text = text.strip()
        if len(text) < self.min_size:
            if text:
                logger.debug("Text below min_size, returning as single chunk", length=len(text))
                return [
                    ChunkResult(
                        chunk_no=0,
                        char_start=0,
                        char_end=len(text),
                        text=text,
                        token_est=max(1, len(text) // 4),
                    )
                ]
            return []

        # Phase 1: split into segments that respect max_size
        segments = self._split_to_segments(text)

        # Phase 2: accumulate segments into target-sized chunks with overlap
        raw_chunks = self._accumulate_with_overlap(text, segments)

        # Phase 3: assign metadata
        results: list[ChunkResult] = []
        for idx, (start, end) in enumerate(raw_chunks):
            chunk_text = text[start:end]
            results.append(
                ChunkResult(
                    chunk_no=idx,
                    char_start=start,
                    char_end=end,
                    text=chunk_text,
                    token_est=max(1, len(chunk_text) // 4),
                )
            )

        logger.info(
            "Chunking complete",
            text_len=len(text),
            chunks=len(results),
            overlap=self.overlap_chars,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_to_segments(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) spans that respect max_size.

        Strategy: split on paragraph boundaries first, then on sentence
        boundaries, then force-split.
        """
        paragraphs = self._split_paragraphs(text)
        segments: list[tuple[int, int]] = []

        for p_start, p_end in paragraphs:
            length = p_end - p_start
            if length <= self.max_size:
                segments.append((p_start, p_end))
            else:
                # Paragraph too large — split on sentence boundaries
                sentences = self._split_sentences(text, p_start, p_end)
                for s_start, s_end in sentences:
                    s_len = s_end - s_start
                    if s_len <= self.max_size:
                        segments.append((s_start, s_end))
                    else:
                        # Force-split at max_size
                        segments.extend(self._force_split(s_start, s_end))

        return segments

    def _split_paragraphs(self, text: str) -> list[tuple[int, int]]:
        """Split text on blank-line boundaries (``\\n\\n``)."""
        parts: list[tuple[int, int]] = []
        pos = 0
        for match in re.finditer(r"\n\n+", text):
            if match.start() > pos:
                parts.append((pos, match.start()))
            pos = match.end()
        if pos < len(text):
            parts.append((pos, len(text)))
        return parts

    def _split_sentences(
        self, text: str, start: int, end: int
    ) -> list[tuple[int, int]]:
        """Split a span on sentence boundaries."""
        sub = text[start:end]
        spans: list[tuple[int, int]] = []
        prev = 0
        for m in _SENTENCE_BOUNDARY.finditer(sub):
            boundary = m.start()
            if boundary > prev:
                spans.append((start + prev, start + boundary))
            prev = m.end()
        if prev < len(sub):
            spans.append((start + prev, start + len(sub)))
        if not spans:
            spans.append((start, end))
        return spans

    def _force_split(self, start: int, end: int) -> list[tuple[int, int]]:
        """Force-split a span into max_size pieces."""
        spans: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            chunk_end = min(pos + self.max_size, end)
            spans.append((pos, chunk_end))
            pos = chunk_end
        return spans

    def _accumulate_with_overlap(
        self,
        text: str,
        segments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Accumulate segments into chunks of ~target_size with overlap.

        Each chunk (except the first) starts overlap_chars before the end
        of the previous chunk, so consecutive chunks share that many
        characters.
        """
        if not segments:
            return []

        chunks: list[tuple[int, int]] = []
        buf_start = segments[0][0]
        buf_end = segments[0][1]

        for seg_start, seg_end in segments[1:]:
            # Would adding this segment exceed the target?
            combined_len = seg_end - buf_start
            if combined_len <= self.target_size:
                # Keep accumulating — include any gap between segments
                buf_end = seg_end
            else:
                # Finalise current chunk
                chunks.append((buf_start, buf_end))

                # Start next chunk with overlap from the end of previous chunk
                overlap_start = max(buf_end - self.overlap_chars, buf_start)
                # The new chunk begins at the overlap point
                buf_start = overlap_start
                buf_end = seg_end

        # Final chunk
        if buf_start < buf_end:
            chunks.append((buf_start, buf_end))

        # Post-process: merge trailing runt chunks (< min_size) into previous
        if len(chunks) > 1:
            last_start, last_end = chunks[-1]
            if (last_end - last_start) < self.min_size:
                prev_start, _prev_end = chunks[-2]
                chunks[-2] = (prev_start, last_end)
                chunks.pop()

        return chunks
