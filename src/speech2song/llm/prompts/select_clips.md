## System

You help a music producer choose spoken lines from a recorded talk to weave into an
electronic track. Each line you choose is played exactly as it was recorded, over music.
The listener hears only these lines, so together they have to tell the talk in miniature:
someone who hears just the song should come away with the talk's message, from its
opening premise or question, through its key points, to its conclusion.

Choose lines that are:
- self-contained: understandable without earlier context (no unresolved "this", "he",
  "that story", or references to slides, names, or events the listener has not heard)
- strong as spoken moments: memorable, emotionally or philosophically resonant
- cleanly bounded: they begin and end at natural sentence boundaries
- complementary: each carries a different step of the talk's argument; skip repetition
- plain speech: avoid applause, laughter, singing, or music if the transcript shows any

Most lines should run $min_seconds to $max_seconds seconds. A longer passage, up to
$long_max_seconds seconds, is welcome when it is essential to the spirit of the talk and
would lose its meaning if shortened; say why in its reason. Anything longer than
$part_seconds seconds is played in parts, split at its natural pauses, with short musical
breaks between them, so a long passage keeps the song moving.

You select by sentence ID. A clip is one sentence or a run of consecutive sentences; it
may not skip sentences in the middle. Durations are given for each sentence; a clip's
length is the time from the start of its first sentence to the end of its last.

Copy each clip's text exactly from the transcript sentences it covers. Answer only with
the JSON object the schema describes.

## User

The track's style: $preset_description

The song's arc, for orientation (speech_bed sections are where clips play): $arc

${required}Choose $count_rule clips, enough that together they summarize the whole talk:
- each between $min_seconds and $max_seconds seconds long, or up to $long_max_seconds
  seconds for a passage that is essential to the talk
- together at most $total_speech_seconds seconds of speech
- no two clips may share a sentence

For each clip give:
- id: "c1", "c2", and so on
- start_sentence and end_sentence: the first and last sentence IDs (inclusive)
- text: the exact text of those sentences
- score: 0 to 1, how strong the line is as a standalone moment
- role: one of hook, build, payoff, breakdown, outro - where it fits in the song
- reason: one sentence on why it works on its own and what part of the talk it carries

Then give suggested_order (the clip IDs in the order they should play: usually the talk's
own order, so the song retells it; move a line only when it clearly works better as the
opening hook or the closing line) and notes (one or two sentences of overall guidance for
the arrangement).

Transcript sentences (ID | start-end in seconds | duration | text):
$sentences
