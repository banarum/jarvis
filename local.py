"""The same call page and tools, but without Gemini Live doing the thinking.

Two pipelines share one brain, any OpenAI-compatible chat model (default:
DeepSeek via opencode Zen):

    local   ears: Whisper on this Mac (mlx-whisper)   mouth: macOS `say`
    speak   ears: Gemini live transcription           mouth: Gemini speech synthesis

server.py uses it when a call asks for the "local" pipeline, or when Gemini
refuses a call over billing. Needs `pip install mlx-whisper` and
JARVIS_BRAIN_KEY in ~/.config/jarvis/env (optionally JARVIS_BRAIN_MODEL,
JARVIS_BRAIN_URL, JARVIS_WHISPER, JARVIS_SAY_VOICE).
"""

import asyncio
import json
import os
import re
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import mlx_whisper
import numpy as np
from aiohttp import ClientSession, ClientTimeout, WSMsgType
from google.genai import types

import server
from server import CFG, CALLS, LOG_DIR, TOOLS, log

BRAIN_MODEL = os.environ.get("JARVIS_BRAIN_MODEL", "deepseek-v4.1-flash")
BRAIN_URL = os.environ.get("JARVIS_BRAIN_URL", "https://opencode.ai/zen/v1").rstrip("/")
BRAIN_KEY = os.environ.get("JARVIS_BRAIN_KEY", "")
WHISPER = os.environ.get("JARVIS_WHISPER", "mlx-community/whisper-turbo")
SAY_VOICE = os.environ.get("JARVIS_SAY_VOICE", "")   # empty = pick by the text's alphabet
EARS_MODEL = os.environ.get("JARVIS_EARS_MODEL", "gemini-3.5-transcribe-live")
MOUTH_MODEL = os.environ.get("JARVIS_MOUTH_MODEL", "gemini-3.8-flash-tts")

SPEED = float(CFG.get("speed") or 1.0)   # speech rate, 1.0 = the voice's own pace
SAY_WPM = 175                            # what macOS voices speak at by default


LOUDNESS = float(CFG.get("loudness") or -16)   # target level of his voice, LUFS
LEVEL_WINDOW = 48000                           # judge a sentence's level from its first second...


async def level(chunks):
    """Bring a sentence to the target loudness. The gain is set from the
    first second and keeps following the sentence's average level, so quiet
    generations are lifted and loud ones tamed without pumping."""
    head, energy, voiced, gain = b"", 0.0, 0, None
    async for c in chunks:
        e, n = voiced_energy(c)
        energy, voiced = energy + e, voiced + n
        if gain is None:
            head += c
            if len(head) < LEVEL_WINDOW:
                continue
            c = head
        want = wanted_gain(energy, voiced)
        gain = want if gain is None else gain * float(np.clip(want / gain, 0.89, 1.12))   # ...then drift at most 1 dB a chunk
        yield apply_gain(c, gain)
    if gain is None and head:   # a sentence shorter than the window
        yield apply_gain(head, wanted_gain(energy, voiced))


def voiced_energy(pcm):
    """Sum of squares and sample count over the 20 ms frames that carry voice."""
    x = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768
    frames = x[:len(x) // 480 * 480].reshape(-1, 480)
    if not len(frames):
        return 0.0, 0
    power = (frames ** 2).mean(axis=1)
    loud = power > 0.003 ** 2        # skip pauses and breaths, keep whispers
    return float(power[loud].sum() * 480), int(loud.sum() * 480)


def wanted_gain(energy, voiced):
    if not voiced:
        return 1.0
    # Speech measures about 1.5 dB quieter in LUFS than its plain RMS level.
    return float(np.clip(10 ** ((LOUDNESS + 1.5) / 20) / np.sqrt(energy / voiced), 0.25, 16.0))


def apply_gain(pcm, gain):
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * gain
    over = np.abs(x) > 26000                                           # soft knee instead of clipping
    x[over] = np.sign(x[over]) * (26000 + 6700 * np.tanh((np.abs(x[over]) - 26000) / 6700))
    return x.astype(np.int16).tobytes()


async def faster(chunks):
    """Speed up a stream of 24 kHz PCM16 chunks by SPEED without raising the pitch."""
    if abs(SPEED - 1) < 0.01:
        async for c in chunks:
            yield c
        return
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error", "-probesize", "32", "-analyzeduration", "0", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
        "-af", f"atempo={SPEED}", "-f", "s16le", "-ar", "24000", "-ac", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)

    async def feed():
        async for c in chunks:
            proc.stdin.write(c)
            await proc.stdin.drain()
        proc.stdin.close()
    feeder = asyncio.create_task(feed())
    carry = b""   # keep chunks on whole 16-bit samples
    try:
        while out := await proc.stdout.read(9600):
            out = carry + out
            carry = out[len(out) // 2 * 2:]
            yield out[:len(out) // 2 * 2]
        await feeder
    finally:
        feeder.cancel()
        if proc.returncode is None:
            proc.kill()


EARS = ThreadPoolExecutor(1)   # one transcription at a time
# What Whisper invents when it is handed silence or noise.
GHOSTS = re.compile(r"субтитр|продолжение следует|спасибо за просмотр|редактор субтитров|thanks for watching|"
                    r"subtitles by|dimatorzok", re.I)

NOTE = """

How this call is wired: {user}'s speech is transcribed by Whisper and handed \
to you as text; your reply is read aloud word for word by a speech \
synthesiser. So reply with speech only: no markdown, no lists, no emoji, \
numbers and abbreviations the way they are said. The transcript can be \
mangled: trust the meaning, not the spelling. If it is clearly noise or not \
addressed to you, reply with exactly: ... (three dots) and nothing is said. \
There is no google_search here; use research for anything from the web.

Your notes from earlier calls:
{notes}"""


def vocabulary():
    """Names the ears should expect: recognisers drop foreign words they weren't told about."""
    return sorted({CFG["assistant_name"], "herdr", "Claude Code", "opencode", *CFG["glossary"],
                   *(a["project"] for a in server.AGENTS.values())} - {""})


def transcribe(pcm):
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    r = mlx_whisper.transcribe(
        audio, path_or_hf_repo=WHISPER, language=CFG["languages"][0], condition_on_previous_text=False,
        initial_prompt=", ".join(vocabulary()[:60]))
    text = " ".join(s["text"].strip() for s in r["segments"] if s.get("no_speech_prob", 0) < 0.6).strip()
    words = text.lower().split()
    looped = len(words) > 6 and len(set(words)) < len(words) * 0.4   # "mb, mb, mb, ..." on near-silence
    return "" if looped or GHOSTS.search(text) else text


def speak(text):
    """Text -> 24 kHz mono PCM16, via the macOS voice."""
    voice = SAY_VOICE or ("Milena" if re.search("[а-яё]", text, re.I) else "Samantha")
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        os.spawnlp(os.P_WAIT, "say", "say", "-v", voice, "-r", str(round(SAY_WPM * SPEED)),
                   "-o", f.name, "--data-format=LEI16@24000", text)
        with wave.open(f.name) as w:
            return w.readframes(w.getnframes())


def openai_tools():
    def low(x):
        if isinstance(x, dict):
            return {k: v.lower() if k == "type" else low(v) for k, v in x.items()}
        return [low(i) for i in x] if isinstance(x, list) else x
    return [{"type": "function", "function": {
        "name": d["name"], "description": d["description"],
        "parameters": low(d.get("parameters") or {"type": "OBJECT", "properties": {}}),
    }} for d in TOOLS[0]["function_declarations"]]


async def chat(messages, tools):
    async with ClientSession(timeout=ClientTimeout(total=180)) as http:
        async with http.post(BRAIN_URL + "/chat/completions", headers={"Authorization": "Bearer " + BRAIN_KEY},
                             json={"model": BRAIN_MODEL, "messages": messages, "tools": tools}) as r:
            data = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"brain model: HTTP {r.status} {str(data)[:300]}")
    u = data.get("usage") or {}
    with open(LOG_DIR / "usage.jsonl", "a") as f:
        f.write(json.dumps({"t": round(time.time(), 2), "kind": "brain", "usd": u.get("cost"),
                            "in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}) + "\n")
    return data["choices"][0]["message"]


class LocalCall(server.Call):
    mode = "local"

    def __init__(self, ws, client):
        super().__init__(ws, client)
        self.msgs = [{"role": "system", "content": server.system_prompt() + NOTE.format(
            user=CFG["user_name"], notes=self.notes())}]
        self.inbox = []                 # context the brain has not seen yet
        self.turns = asyncio.Lock()     # one exchange at a time
        self.audio = bytearray()        # the utterance being heard
        self.loud = 0                   # loud frames in it, to tell speech from a click
        self.talking = None             # task that is speaking right now
        self.session = self             # the base class only checks that a session exists

    # --- wiring the base class expects ---

    async def run(self):
        for old in list(CALLS):
            old.closed = True
            await old.ws.close()
        CALLS.add(self)
        try:
            self.log("connected", resumed=False, pipeline="local")
            await self.to_browser(type="status", text="connected")
            await self.to_browser(type="mode", mode=self.mode)
            asyncio.create_task(self.open_call(self))
            ears = asyncio.create_task(self.listen())
            await self.read_browser()
        finally:
            self.closed = True
            ears.cancel()
            CALLS.discard(self)
            self.log("hangup", seconds=round(time.time() - self.started))
            if not self.ws.closed:
                await self.ws.close()
            self.logf.close()

    async def say(self, s, text, turn_complete=True):
        """System messages from the base class (briefing, greeting, agent news)."""
        if turn_complete:
            asyncio.create_task(self.turn(text))
        else:
            self.inbox.append(text)

    # --- ears ---

    async def read_browser(self):
        async for msg in self.ws:
            if msg.type == WSMsgType.BINARY:
                now = time.time()
                if now - self.last_in > 1.5:
                    self.log("speech_start")
                self.last_in = now
                await self.hear(msg.data)
                pcm = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768
                self.loud += float(np.sqrt(np.mean(pcm ** 2))) > 0.02
                if self.loud == 8 and self.talking and not self.talking.done():   # barge-in
                    self.talking.cancel()
                    await self.to_browser(type="interrupted")
            elif msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("type") == "confirm":
                    asyncio.create_task(self.tapped(bool(m.get("ok"))))
                elif m.get("type") == "pause":
                    loud, self.loud = self.loud, 0
                    await self.pause(loud)
                elif m.get("type") == "text" and m.get("text"):
                    asyncio.create_task(self.user_said(m["text"]))
        self.closed = True

    async def listen(self):
        """Background work the ears need for the whole call. Whisper needs none."""

    async def hear(self, frame):
        self.audio += frame

    async def pause(self, loud):
        """The caller went quiet: transcribe what was said."""
        pcm, self.audio = bytes(self.audio), bytearray()
        if loud >= 5:
            asyncio.create_task(self.heard_audio(pcm))

    async def heard_audio(self, pcm):
        t0 = time.time()
        text = await asyncio.get_running_loop().run_in_executor(EARS, transcribe, pcm)
        self.log("ears", seconds=round(len(pcm) / 32000, 1), ms=int((time.time() - t0) * 1000), text=text)
        if text:
            await self.user_said(text)

    async def user_said(self, text):
        self.last_user_at = time.time()
        self.heard.append((self.last_user_at, text))
        del self.heard[:-50]
        await self.to_browser(type="you", text=text)
        await self.to_browser(type="turn_complete")
        self.log("you", text=text)
        await self.turn(text)

    # --- brain ---

    async def turn(self, text):
        async with self.turns:
            if self.closed:
                return
            t0 = time.time()
            await self.to_browser(type="tool", text="thinking")   # starts the tick
            try:
                reply = await self.think(text)
            finally:
                await self.to_browser(type="tool_done")
            self.log("brain", ms=int((time.time() - t0) * 1000), say=reply)
            if reply and reply.strip(" .…"):
                self.talking = asyncio.create_task(self.speak(reply))
                try:
                    await self.talking
                except asyncio.CancelledError:
                    self.log("interrupted")

    async def think(self, text):
        m = self.msgs
        m.extend({"role": "user", "content": t} for t in self.inbox)
        self.inbox.clear()
        m.append({"role": "user", "content": text})
        tools, msg = openai_tools(), {}
        try:
            for _ in range(10):
                msg = await chat(m, tools)
                m.append({k: v for k, v in msg.items() if v is not None})
                calls = msg.get("tool_calls") or []
                if not calls:
                    break
                results = await asyncio.gather(*(self.brain_tool(c) for c in calls))
                m.extend({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(r, ensure_ascii=False)}
                         for c, r in zip(calls, results))
        except Exception as e:
            log.exception("brain failed")
            self.log("brain_error", error=str(e))
            del m[1:]   # a half-finished exchange would poison every later request
            return "Мозг не отвечает." if CFG["languages"][0] == "ru" else "The brain is not answering."
        # Forget the oldest exchanges once the history gets long.
        while sum(len(str(x.get("content") or "")) for x in m) > 200_000 and len(m) > 3:
            del m[1:next((i for i in range(2, len(m)) if m[i]["role"] == "user"), len(m))]
        return msg.get("content") or ""

    async def brain_tool(self, call):
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except ValueError:
            return {"error": "arguments were not valid JSON"}
        return await self.call_tool(call["function"]["name"], args)

    # --- mouth ---

    async def speak(self, text):
        await self.to_browser(type="jarvis", text=text)
        self.log("jarvis", text=text)
        ends = time.time()   # when the browser will have played everything sent so far
        try:
            for sentence in re.findall(r".+?(?:[.!?…]+(?=\s|$)|$)", text.replace("\n", " ")):
                if not sentence.strip():
                    continue
                async for pcm in self.voice(sentence.strip()):
                    if self.ws.closed:
                        return
                    await self.ws.send_bytes(pcm)
                    ends = max(ends, time.time()) + len(pcm) / 48000
                    self.last_out = ends
            # Stay "talking" while the browser plays it, so a barge-in can cut it.
            await asyncio.sleep(max(0, ends - time.time() - 0.3))
        finally:
            await self.to_browser(type="turn_complete")

    def voice(self, sentence):
        """One sentence as 24 kHz mono PCM16 chunks, at an even loudness."""
        return level(self.synth(sentence))

    async def synth(self, sentence):
        yield await asyncio.get_running_loop().run_in_executor(None, speak, sentence)


class SpeakCall(LocalCall):
    """Gemini as ears and mouth only; the brain model does the thinking."""
    mode = "speak"

    def __init__(self, ws, client):
        super().__init__(ws, client)
        self.ears = None   # live transcription session

    async def listen(self):
        config = types.LiveConnectConfig(
            input_audio_transcription=types.AudioTranscriptionConfig(custom_vocabulary=vocabulary()[:100]))
        while not self.closed:   # sessions last ~10 minutes; each utterance stands alone, so just reconnect
            try:
                async with self.client.aio.live.connect(model=EARS_MODEL, config=config) as s:
                    self.ears = s
                    while not self.closed:
                        async for r in s.receive():
                            if r.usage_metadata:
                                server.spend("ears", r.usage_metadata)
                            if r.go_away is not None:
                                raise ConnectionError("go_away")
                            sc = r.server_content
                            if sc and sc.input_transcription and sc.input_transcription.text:
                                self.log("ears", text=sc.input_transcription.text)
                                asyncio.create_task(self.user_said(sc.input_transcription.text.strip()))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self.closed:
                    self.log("ears_error", error=str(e)[:200])
                    await asyncio.sleep(0.5)
            finally:
                self.ears = None

    async def hear(self, frame):
        if self.ears:
            await self.ears.send_realtime_input(audio=types.Blob(data=frame, mime_type="audio/pcm;rate=16000"))

    async def pause(self, loud):
        # The page stops sending silence; tell the transcriber the phrase is over.
        if self.ears:
            await self.ears.send_realtime_input(audio_stream_end=True)

    def voice(self, sentence):
        return faster(level(self.synth(sentence)))   # macOS `say` has its own rate; Gemini's voice is sped up here

    async def synth(self, sentence):
        config = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=CFG["voice"]))))
        usage = None
        async for r in await self.client.aio.models.generate_content_stream(
                model=MOUTH_MODEL, contents=sentence, config=config):
            usage = r.usage_metadata or usage
            for cand in r.candidates or []:
                for part in (cand.content.parts if cand.content and cand.content.parts else []):
                    if part.inline_data and part.inline_data.data:
                        yield part.inline_data.data
        server.spend("mouth", usage)


MODES = {"local": LocalCall, "speak": SpeakCall}

