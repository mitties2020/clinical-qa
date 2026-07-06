(function () {
  "use strict";

  const chunkMs = Number(window.MIC_TRANSCRIBE_CHUNK_MS || 8000);
  const micState = { status: "idle", message: "" };
  const callState = { status: "idle", message: "" };

  function text(value) {
    return String(value || "").trim();
  }

  function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
  }

  function appendText(value) {
    const clean = text(value);
    if (!clean) return;
    if (typeof window.appendTranscriptToConsultInput === "function") {
      window.appendTranscriptToConsultInput(clean);
      return;
    }
    const input = document.getElementById("clinicalInput");
    if (!input) return;
    const spacer = input.value ? (input.value.endsWith("\n") ? "" : "\n\n") : "";
    input.value = `${input.value}${spacer}${clean}`;
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function renderTranscriptStatus() {
    if (micState.status === "requesting") {
      setText("transcriptionStatus", micState.message || "Requesting microphone permission...");
      return;
    }
    if (micState.status === "active") {
      setText("transcriptionStatus", micState.message || "Mic dictation active");
      return;
    }
    if (micState.status === "processing") {
      setText("transcriptionStatus", micState.message || "Transcribing mic audio...");
      return;
    }
    if (micState.status === "error") {
      setText("transcriptionStatus", micState.message || "Mic transcription failed");
      return;
    }
    if (micState.status === "ready") {
      setText("transcriptionStatus", micState.message || "Mic transcription ready");
      return;
    }
    if (callState.status === "connected") {
      setText("transcriptionStatus", callState.message || "Call transcript connected; waiting for audio");
      return;
    }
    if (callState.status === "active") {
      setText("transcriptionStatus", callState.message || "Call transcription active");
      return;
    }
    if (callState.status === "stream" || callState.status === "audio") {
      setText("transcriptionStatus", callState.message || "Receiving call audio");
      return;
    }
    if (callState.status === "preview") {
      setText("transcriptionStatus", callState.message || "Hearing call audio...");
      return;
    }
    if (callState.status === "error") {
      setText("transcriptionStatus", callState.message || "Call transcription unavailable");
      return;
    }
    setText("transcriptionStatus", "Call transcription idle");
  }

  function setMicStatus(status, message) {
    const next = text(status) || "inactive";
    const msg = text(message);
    if (next === "active") setText("micStatus", msg || "Mic dictation active");
    else if (next === "requesting") setText("micStatus", msg || "Requesting microphone permission...");
    else if (next === "processing") setText("micStatus", msg || "Transcribing mic audio...");
    else if (next === "ready") setText("micStatus", msg || "Mic transcription added to input");
    else if (next === "unsupported") setText("micStatus", msg || "Mic dictation unsupported");
    else if (next === "error") setText("micStatus", msg || "Mic transcription failed");
    else setText("micStatus", msg || "Mic dictation inactive");

    micState.status = ["active", "requesting", "processing", "ready", "error"].includes(next) ? next : "idle";
    micState.message = msg;
    renderTranscriptStatus();

    if (next === "ready") {
      window.setTimeout(() => {
        if (micState.status === "ready") {
          micState.status = "idle";
          micState.message = "";
          renderTranscriptStatus();
        }
      }, 2500);
    }
  }

  function setTranscriptStatus(status, message) {
    const next = text(status) || "idle";
    callState.status = next === "stopped" || next === "disconnected" ? "idle" : next;
    callState.message = text(message);
    renderTranscriptStatus();
  }

  function setButtons(state) {
    const startBtn = document.getElementById("startMicBtn");
    const stopBtn = document.getElementById("stopMicBtn");
    if (startBtn) startBtn.disabled = state === "active" || state === "processing" || state === "unsupported";
    if (stopBtn) stopBtn.disabled = state !== "active";
  }

  function pickRecorderOptions() {
    if (!window.MediaRecorder || typeof window.MediaRecorder.isTypeSupported !== "function") return {};
    for (const mimeType of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]) {
      if (window.MediaRecorder.isTypeSupported(mimeType)) return { mimeType };
    }
    return {};
  }

  async function deepgramConfigured() {
    if (window.PREFER_BROWSER_SPEECH === true || window.PREFER_SERVER_MIC_TRANSCRIPTION === false) return false;
    if (!window.MediaRecorder || !window.navigator?.mediaDevices?.getUserMedia) return false;
    try {
      const response = await fetch("/api/transcription-health", { credentials: "same-origin" });
      if (!response.ok) return false;
      const data = await response.json();
      return Boolean(data.deepgramConfigured);
    } catch {
      return false;
    }
  }

  function setupSpeechRecognition(SpeechRecognition) {
    const startBtn = document.getElementById("startMicBtn");
    const stopBtn = document.getElementById("stopMicBtn");
    const input = document.getElementById("clinicalInput");
    const recognition = new SpeechRecognition();
    let active = false;
    let baseText = "";
    let restartTimer = null;

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = window.MIC_DICTATION_LANG || "en-AU";

    function stopRestartTimer() {
      if (restartTimer) window.clearTimeout(restartTimer);
      restartTimer = null;
    }

    function startRecognition() {
      try {
        recognition.start();
      } catch {
        restartTimer = window.setTimeout(() => {
          if (active) startRecognition();
        }, 300);
      }
    }

    if (startBtn) {
      startBtn.addEventListener("click", () => {
        if (!input) return;
        stopRestartTimer();
        baseText = input.value;
        active = true;
        setMicStatus("active", "Mic dictation active");
        setButtons("active");
        startRecognition();
      });
    }

    if (stopBtn) {
      stopBtn.addEventListener("click", () => {
        active = false;
        stopRestartTimer();
        try {
          recognition.stop();
        } catch {}
        setMicStatus("inactive");
        setButtons("inactive");
      });
    }

    recognition.onresult = (event) => {
      if (!input) return;
      let finalChunk = "";
      let interimChunk = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const transcript = text(event.results[index][0]?.transcript);
        if (!transcript) continue;
        if (event.results[index].isFinal) finalChunk += `${transcript} `;
        else interimChunk += `${transcript} `;
      }
      const finalText = text(finalChunk);
      const interimText = text(interimChunk);
      let committed = baseText.trimEnd();
      if (finalText) {
        committed = committed ? `${committed} ${finalText}` : finalText;
        baseText = committed;
      }
      input.value = interimText ? `${committed}${committed ? " " : ""}${interimText}` : committed;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      setMicStatus("active", interimText ? "Mic dictation active; listening..." : "Mic dictation active");
    };

    recognition.onend = () => {
      if (active) {
        setMicStatus("active", "Mic dictation active; reconnecting listener...");
        restartTimer = window.setTimeout(startRecognition, 250);
        return;
      }
      setMicStatus("inactive");
      setButtons("inactive");
    };

    recognition.onerror = (event) => {
      if (active && ["no-speech", "aborted"].includes(event.error)) return;
      active = false;
      setMicStatus("error", event.error ? `Mic dictation error: ${event.error}` : "Mic dictation error");
      setButtons("inactive");
    };

    setMicStatus("inactive");
    setButtons("inactive");
  }

  function setupMediaRecorder() {
    const startBtn = document.getElementById("startMicBtn");
    const stopBtn = document.getElementById("stopMicBtn");
    let recorder = null;
    let stream = null;
    let uploadChain = Promise.resolve();
    let stopped = false;
    let addedText = false;

    async function upload(blob) {
      const form = new FormData();
      form.append("audio", blob, "dictation.webm");
      const response = await fetch("/api/transcribe", { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const fallback = response.status === 401
          ? "Please log in again before dictating."
          : `Mic transcription failed (${response.status}).`;
        throw new Error(data.error || fallback);
      }
      return text(data.text);
    }

    function queueUpload(blob) {
      if (!blob || blob.size < 512) return uploadChain;
      uploadChain = uploadChain.catch(() => {}).then(async () => {
        setMicStatus("processing", stopped ? "Finishing mic transcription..." : "Mic dictation active; transcribing recent audio...");
        const transcript = await upload(blob);
        if (transcript) {
          addedText = true;
          appendText(transcript);
          setMicStatus("active", "Mic dictation active; text added");
        }
      }).catch((error) => {
        setMicStatus("error", error.message || "Mic transcription failed");
      });
      return uploadChain;
    }

    if (startBtn) {
      startBtn.addEventListener("click", async () => {
        try {
          stopped = false;
          addedText = false;
          uploadChain = Promise.resolve();
          setMicStatus("requesting");
          setButtons("processing");
          stream = await window.navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
              channelCount: 1,
            },
          });
          recorder = new MediaRecorder(stream, pickRecorderOptions());
          recorder.ondataavailable = (event) => queueUpload(event.data);
          recorder.onstop = async () => {
            stopped = true;
            stream?.getTracks().forEach((track) => track.stop());
            setButtons("processing");
            setMicStatus("processing", "Finishing mic transcription...");
            await uploadChain.catch(() => {});
            recorder = null;
            stream = null;
            setButtons("inactive");
            setMicStatus(addedText ? "ready" : "inactive", addedText ? "Mic transcription added to input" : "No speech detected in mic recording");
          };
          recorder.start(Math.max(3000, chunkMs));
          setMicStatus("active", "Mic dictation active; transcribing in chunks");
          setButtons("active");
        } catch {
          setMicStatus("error", "Microphone permission denied or unavailable");
          setButtons("inactive");
        }
      });
    }

    if (stopBtn) {
      stopBtn.addEventListener("click", () => {
        if (!recorder || recorder.state === "inactive") return;
        try {
          recorder.requestData();
        } catch {}
        recorder.stop();
      });
    }

    setMicStatus("inactive");
    setButtons("inactive");
  }

  window.setMicStatus = setMicStatus;
  window.setTranscriptStatus = setTranscriptStatus;

  window.initMicDictation = async function initMicDictation() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const hasRecorder = Boolean(window.MediaRecorder && window.navigator?.mediaDevices?.getUserMedia);
    setMicStatus("requesting", "Checking mic transcription...");
    if (await deepgramConfigured()) {
      setupMediaRecorder();
      return;
    }
    if (SpeechRecognition) {
      setupSpeechRecognition(SpeechRecognition);
      return;
    }
    if (hasRecorder) {
      setupMediaRecorder();
      return;
    }
    setMicStatus("unsupported");
    setButtons("unsupported");
  };

  window.connectTranscriptSocket = function connectTranscriptSocket() {
    const configured = text(window.TRANSCRIPT_WS_URL || "");
    const disableDefault = Boolean(window.DISABLE_DEFAULT_TRANSCRIPT_WS);
    const allowTunnel = Boolean(window.ALLOW_TUNNEL_WS);
    const wsUrl = configured || (!disableDefault ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/frontend-transcript` : "");
    if (!wsUrl) {
      setTranscriptStatus("stopped");
      return;
    }
    if (/ngrok|tunnel/i.test(wsUrl) && !allowTunnel) {
      setTranscriptStatus("stopped");
      return;
    }
    try {
      const socket = new WebSocket(wsUrl);
      socket.addEventListener("open", () => setTranscriptStatus("connected"));
      socket.addEventListener("close", () => setTranscriptStatus("disconnected"));
      socket.addEventListener("error", () => setTranscriptStatus("error", "Transcript connection unavailable"));
      socket.addEventListener("message", (event) => {
        let data;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }
        if (data?.type === "status") setTranscriptStatus(data.status, data.message);
        if (data?.type === "transcript-preview") setTranscriptStatus("preview", data.text ? `Hearing: ${data.text}` : "Hearing call audio...");
        if (data?.type === "transcript") {
          appendText(data.text);
          setTranscriptStatus("active", "Call transcription active; text added");
        }
      });
    } catch {
      setTranscriptStatus("error", "Transcript connection unavailable");
    }
  };
})();
