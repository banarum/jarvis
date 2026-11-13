"""The turn completion protocol: the brain itself says whether the caller has finished.

An acoustic model hears how a phrase ends; only the language model knows whether
"давай сделаем лидов, то есть" is a finished thought. Every reply starts with one
marker: complete (answer follows), cut off mid-sentence, or needs time. For the last
two nothing is said and the next words are judged together with these.

The protocol text and the two re-prompts below are taken from Pipecat
(pipecat/turns/user_turn_completion_mixin.py), Copyright (c) 2024-2026, Daily,
BSD 2-Clause License: redistribution in source form must retain that notice.
"""

COMPLETE, SHORT, LONG = "\u25cf", "\u25d0", "\u25cb"
WAIT = {SHORT: 5.0, LONG: 10.0}      # seconds of quiet before the brain is asked to speak after all

PROTOCOL = """
TURN COMPLETION PROTOCOL (mandatory):
The user's words reach you from speech recognition, usually without punctuation, and sometimes before they have finished talking. Before you reply, decide whether their turn is complete, and start every response with exactly one of these markers as its very first character:

●  the user's turn is complete: answer them. Write ●, a space, then your full reply. Never write ● on its own.
◐  the user stopped mid-sentence and will continue in a few seconds. Write ◐ and nothing else.
○  the user needs time to think or asked you to wait. Write ○ and nothing else.

Deciding:
- Complete means conversationally complete, not long. One word can be a complete answer to your question: "yes", "no thanks", "Tuesday", "Japan", "four". A question to you is complete. A correction, or a request to repeat yourself, is complete.
- Grammatically complete is not the same as conversationally complete. If the user has only acknowledged your question or reacted to what you said, without answering it ("that's a good question", "oh wow, okay", "that's interesting", "hmm", "well"), they have not taken their turn yet: ○.
- Cut off (◐): the last words leave a phrase open, in whatever language the user speaks: a sentence that ends on a conjunction, a preposition, an article, or the word for "because"; a list that is still going; a number that is only partly said. A fragment like this is not a request for help. Do not answer it, do not ask what they need, and do not guess the rest: wait with ◐.
- Needs time (○): "hold on", "let me think", "give me a second", "one moment". Filler followed by a real answer, such as "hmm, I'd go to Japan for the food", is complete.
- When the user's latest words continue an earlier fragment (your previous response was ◐ or ○), judge the fragments together as one turn. If the combined turn still ends open, ◐; if it now answers your question, ●.
- If a tool call is the right response, make the tool call; the turn is complete.

Format rules:
- The marker is the first character. No text, quotes, backticks or explanation before it.
- After ◐ or ○, output nothing: no words, no explanation, no nudge. The system waits and prompts you again later.
- Exactly one marker per response.

Examples:
- You asked where they would go and the user says "i'd go to japan because i love". Respond with only ◐.
- The user says "i need help with". Respond with only ◐.
- You asked for their phone number and the user says "it's five five five". Respond with only ◐.
- You asked what they want to order and the user says "a large pepperoni pizza a garden salad and". Respond with only ◐.
- You asked where they would go and the user says "that's a good question let me think". Respond with only ○.
- You asked where they would go and the user says "that's interesting". Respond with only ○.
- The user says "hold on a second". Respond with only ○.
- You asked where they would go and the user says "japan". Respond with ● followed by your reply, for example "● Japan is a wonderful choice. What draws you there?"
- You asked whether to book it and the user says "yes". Respond with ● followed by your reply, for example "● Done, I'll book it now."
- The user says "can you help me book a flight to new york next week". Respond with ● followed by your reply, for example "● Of course. What day would you like to leave, and from which city?"
- Your previous response was ◐ after the user said "i'd go to", and now the user says "japan because". Together that is still open: respond with only ◐."""

NUDGE_SHORT = """The user paused briefly. Generate a brief, natural prompt to encourage them to continue.

IMPORTANT: You MUST respond with ● followed by your message. Do NOT output ◐ or ○ - the user has already been given time to continue.

Your response should:
- Be contextually relevant to what was just discussed
- Sound natural and conversational
- Be very concise (1 sentence max)
- Gently prompt them to continue

Example format: ● Go ahead, I'm listening.

Generate your ● response now."""

NUDGE_LONG = """The user has been quiet for a while. Generate a friendly check-in message.

IMPORTANT: You MUST respond with ● followed by your message. Do NOT output ◐ or ○ - the user has already been given plenty of time.

Your response should:
- Acknowledge they might be thinking or busy
- Offer to help or continue when ready
- Be warm and understanding
- Be brief (1 sentence)

Example format: ● No rush! Let me know when you're ready to continue.

Generate your ● response now."""

